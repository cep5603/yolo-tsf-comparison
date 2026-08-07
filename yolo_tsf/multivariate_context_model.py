import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import PatchTST_Embedding, RevIN
from temporal_gating_model import (
    Backbone,
    Neck,
    SeparateMultiScaleForecastHead,
    TemporalPyramidInjection,
)


class CrossChannelMixer(nn.Module):
    def __init__(self, n_vars, hidden=64):
        super().__init__()
        self.channel_mix = nn.Sequential(
            nn.Conv1d(n_vars, hidden, 1),
            nn.SiLU(),
            nn.Conv1d(hidden, n_vars, 1),
        )
        self.temporal_mix = nn.Sequential(
            nn.Conv1d(n_vars, n_vars, 3, padding=1, groups=n_vars),
            nn.SiLU(),
            nn.Conv1d(n_vars, n_vars, 1),
        )
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        mixed = self.channel_mix(x) + self.temporal_mix(x)
        return x + self.alpha * mixed


class MultivariateStatsEncoder(nn.Module):
    def __init__(self, n_vars, hidden=64, eps=1e-5):
        super().__init__()
        self.n_vars = n_vars
        self.eps = eps
        self.net = nn.Sequential(
            nn.Linear(8 * n_vars, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )

    def forward(self, x):
        length = x.shape[-1]
        short_len = max(2, length // 8)
        short = x[..., -short_len:]
        mean = x.mean(dim=-1)
        std = x.var(dim=-1, unbiased=False).add(self.eps).sqrt()
        last = x[..., -1]
        first = x[..., 0]
        short_mean = short.mean(dim=-1)
        short_std = short.var(dim=-1, unbiased=False).add(self.eps).sqrt()
        short_first = short[..., 0]
        short_last = short[..., -1]
        denom = std + self.eps
        stats = torch.stack(
            [
                (mean - last) / denom,
                (short_mean - last) / denom,
                (first - last) / denom,
                (short_last - short_first) / denom / max(1, short_len - 1),
                (last - first) / denom / max(1, length - 1),
                short_std / denom,
                torch.log(std + self.eps),
                torch.log(short_std + self.eps),
            ],
            dim=-1,
        )
        return self.net(stats.flatten(start_dim=1))


class MultivariateSmoothTrendHead(nn.Module):
    def __init__(self, channels, stats_dim, horizon, n_vars, controls=8):
        super().__init__()
        self.horizon = horizon
        self.n_vars = n_vars
        self.controls = min(controls, horizon)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.net = nn.Sequential(
            nn.Linear(channels + stats_dim, 128),
            nn.SiLU(),
            nn.Linear(128, self.controls * n_vars),
        )
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, p5, stats):
        pooled = self.pool(p5).squeeze(-1)
        control = self.net(torch.cat([pooled, stats], dim=1))
        control = control.view(control.shape[0], self.n_vars, self.controls)
        trend = F.interpolate(
            control,
            size=self.horizon,
            mode="linear",
            align_corners=self.controls > 1,
        )
        return self.scale * trend.transpose(1, 2)


class MultivariateHorizonFusion(nn.Module):
    def __init__(self, horizon, n_vars, stats_dim, channel_dim=8, hidden=64):
        super().__init__()
        self.horizon = horizon
        self.n_vars = n_vars
        self.channel_embed = nn.Parameter(torch.zeros(1, 1, n_vars, channel_dim))
        nn.init.trunc_normal_(self.channel_embed, std=0.02)
        self.net = nn.Sequential(
            nn.Linear(stats_dim + channel_dim + 3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, f3, f4, stats):
        batch = f3.shape[0]
        pos = torch.linspace(0, 1, self.horizon, device=f3.device, dtype=f3.dtype).view(1, self.horizon, 1, 1)
        pos = pos.expand(batch, -1, self.n_vars, -1)
        values = torch.stack([f3, f4], dim=-1)
        stats = stats.view(batch, 1, 1, -1).expand(-1, self.horizon, self.n_vars, -1)
        channels = self.channel_embed.to(dtype=f3.dtype).expand(batch, self.horizon, -1, -1)
        fused = self.net(torch.cat([values, pos, channels, stats], dim=-1)).squeeze(-1)
        return fused


class YOLO11_TSF(nn.Module):
    def __init__(
        self,
        horizon=10,
        use_skip=True,
        use_revin=True,
        seq_len=96,
        patch_len=16,
        patch_stride=8,
        n_vars=1,
        return_components=False,
    ):
        super().__init__()
        self.horizon = horizon
        self.n_vars = n_vars
        self.use_skip = use_skip
        self.use_revin = use_revin
        self.return_components = return_components
        self.stats_dim = 64
        self.width = [32 * n_vars, 64, 128, 256, 256]

        if use_revin:
            self.revin = RevIN(num_features=n_vars, affine=True)

        self.input_mixer = CrossChannelMixer(n_vars=n_vars)
        self.stats_encoder = MultivariateStatsEncoder(n_vars=n_vars, hidden=self.stats_dim)
        self.patch = PatchTST_Embedding(
            seq_len=seq_len,
            patch_len=patch_len,
            stride=patch_stride,
            d_model=32,
        )
        self.backbone = Backbone(width=self.width)
        self.temporal_injection = TemporalPyramidInjection(
            patch_channels=self.width[0],
            channels=[128, 256, 256],
        )
        self.neck = Neck(width=self.width)
        self.head = SeparateMultiScaleForecastHead(
            channels=[128, 256, 256],
            horizon=horizon * n_vars,
        )
        self.trend_head = MultivariateSmoothTrendHead(
            channels=256,
            stats_dim=self.stats_dim,
            horizon=horizon,
            n_vars=n_vars,
        )
        self.fusion = MultivariateHorizonFusion(
            horizon=horizon,
            n_vars=n_vars,
            stats_dim=self.stats_dim,
        )
        self.log_sigma_p3 = nn.Parameter(torch.zeros(1))
        self.log_sigma_p4 = nn.Parameter(torch.zeros(1))
        self.log_sigma_p5 = nn.Parameter(torch.zeros(1))

    def slope_loss(self, pred, target):
        if pred.shape[-1] < 2:
            return torch.zeros((), dtype=pred.dtype, device=pred.device)
        pred_delta = pred[..., 1:] - pred[..., :-1]
        target_delta = target[..., 1:] - target[..., :-1]
        return F.mse_loss(pred_delta, target_delta)

    def curvature_loss(self, pred, target):
        if pred.shape[-1] < 3:
            return torch.zeros((), dtype=pred.dtype, device=pred.device)
        pred_curve = pred[..., 2:] - 2.0 * pred[..., 1:-1] + pred[..., :-2]
        target_curve = target[..., 2:] - 2.0 * target[..., 1:-1] + target[..., :-2]
        return F.mse_loss(pred_curve, target_curve)

    def _to_channel_first(self, x):
        if x.ndim != 3:
            raise ValueError(f"Expected 3D input, got shape={tuple(x.shape)}")
        if x.shape[1] == self.n_vars:
            return x
        if x.shape[2] == self.n_vars:
            return x.transpose(1, 2)
        raise ValueError(f"Expected input with n_vars={self.n_vars}, got shape={tuple(x.shape)}")

    def _reshape_head(self, y):
        return y.view(y.shape[0], self.n_vars, self.horizon).transpose(1, 2)

    def forward(self, x, return_components=None):
        if return_components is None:
            return_components = self.return_components

        x = self._to_channel_first(x)
        raw_x = x
        stats = self.stats_encoder(raw_x)
        last_val = x[:, :, -1:] if self.use_skip else 0

        if self.use_revin:
            x = self.revin(x, mode="norm")

        x = self.input_mixer(x)
        z = self.patch(x)
        p3, p4, p5 = self.backbone(z)
        p3, p4, p5 = self.temporal_injection(z, p3, p4, p5)
        feats = self.neck(p3, p4, p5)
        hp3, hp4, hp5 = [self._reshape_head(y) for y in self.head(feats)]
        trend_delta = self.trend_head(feats[2], stats)
        hp5 = hp5 + trend_delta
        detail = self.fusion(hp3, hp4, stats)
        detail = detail - detail.mean(dim=1, keepdim=True)
        out = detail + hp5

        out_cf = out.transpose(1, 2)
        hp3_cf = hp3.transpose(1, 2)
        hp4_cf = hp4.transpose(1, 2)
        hp5_cf = hp5.transpose(1, 2)

        if self.use_revin:
            mode = "denorm_delta" if self.use_skip else "denorm"
            out_cf = self.revin(out_cf, mode=mode)
            hp3_cf = self.revin(hp3_cf, mode=mode)
            hp4_cf = self.revin(hp4_cf, mode=mode)
            hp5_cf = self.revin(hp5_cf, mode=mode)
            if out_cf.ndim == 2:
                out_cf = out_cf.unsqueeze(1)
                hp3_cf = hp3_cf.unsqueeze(1)
                hp4_cf = hp4_cf.unsqueeze(1)
                hp5_cf = hp5_cf.unsqueeze(1)

        if self.use_skip:
            out_cf = out_cf + last_val

        out = out_cf.transpose(1, 2)
        hp3 = hp3_cf.transpose(1, 2)
        hp4 = hp4_cf.transpose(1, 2)
        hp5 = hp5_cf.transpose(1, 2)

        if return_components:
            return out, hp3, hp4, hp5

        return out
