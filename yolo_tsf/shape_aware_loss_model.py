import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import PatchTST_Embedding, RevIN

"""
This version tries to improve performance by introducing

* a shape-aware loss for p5
* dynamic loss weighting
"""

class Conv1D(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None):
        super().__init__()
        self.conv = nn.Conv1d(c1, c2, k, s, padding=k//2 if p is None else p, bias=False)
        self.bn = nn.BatchNorm1d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class C3K2_1D(nn.Module):
    def __init__(self, c1, c2, n=2):
        super().__init__()
        c_ = c2 // 2

        self.cv1 = Conv1D(c1, c_, 1)
        self.cv2 = Conv1D(c1, c_, 1)

        self.m = nn.Sequential(*[
            nn.Sequential(
                Conv1D(c_, c_, 1),
                Conv1D(c_, c_, 3)
            ) for _ in range(n)
        ])

        self.cv3 = Conv1D(c2, c2, 1)

    def forward(self, x):
        y1 = self.cv1(x)
        y1 = self.m(y1)
        y2 = self.cv2(x)
        return self.cv3(torch.cat([y1, y2], dim=1))

class SPPF1D(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        self.cv1 = Conv1D(c1, c2, 1)
        #self.pool = nn.MaxPool1d(kernel_size=k, stride=1, padding=k // 2)
        self.pool = nn.AvgPool1d(kernel_size=k, stride=1, padding=k // 2)
        self.cv2 = Conv1D(c2 * 4, c2, 1)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))

# x: (batch, channels=c, length=L)
class C2PSA1D(nn.Module):
    def __init__(self, c, reduction=4):
        super().__init__()
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), #global average pooling over the temporal dimension, (batch, c, 1)
            nn.Conv1d(c, c // reduction, 1), #Reduces channel dimension and works like a fully-connected layer across channels
            nn.SiLU(),
            nn.Conv1d(c // reduction, c, 1), #Expands back to original number of channels (batch, c, 1)
            nn.Sigmoid()
        )

        self.spatial_attn = nn.Sequential(
            nn.Conv1d(c, 1, kernel_size=7, padding=3), #This examines the multi-channel feature map and produces 1 attention weight per time step
            nn.Sigmoid() #(batch, 1, L)
        )

    def forward(self, x):
        ca = self.channel_attn(x)
        sa = self.spatial_attn(x)
        return x * ca * sa

class PatchEmbed1D(nn.Module):
    def __init__(self, in_ch, embed_dim, patch_len, stride):
        super().__init__()
        self.proj = nn.Conv1d(in_ch, embed_dim, patch_len, stride)

    def forward(self, x):
        return self.proj(x)

class Backbone(nn.Module):
    def __init__(self, width):
        super().__init__()

        self.p2 = nn.Sequential(
            Conv1D(width[0], width[1], 3, 2),
            C3K2_1D(width[1], width[1])
        )

        self.p3 = nn.Sequential(
            Conv1D(width[1], width[2], 3, 2),
            C3K2_1D(width[2], width[2])
        )

        self.p4 = nn.Sequential(
            Conv1D(width[2], width[3], 3, 2),
            C3K2_1D(width[3], width[3])
        )

        self.p5 = nn.Sequential(
            Conv1D(width[3], width[4], 3, 2),
            C3K2_1D(width[4], width[4]),
            SPPF1D(width[4], width[4]),
            C2PSA1D(width[4])
        )

    def forward(self, x):
        p2 = self.p2(x)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return p3, p4, p5

class UpsampleLike1D(nn.Module):
    def forward(self, src, target):
        return F.interpolate(
            src,
            size=target.shape[-1],
            mode="nearest"
        )

class Neck(nn.Module):
    def __init__(self, width):
        super().__init__()

        self.up = UpsampleLike1D()

        # Top-down
        self.reduce_p5 = Conv1D(width[4], width[3], 1)
        self.fpn_p4 = C3K2_1D(width[3] * 2, width[3])

        self.reduce_p4 = Conv1D(width[3], width[2], 1)
        self.fpn_p3 = C3K2_1D(width[2] * 2, width[2])

        # Bottom-up
        self.down_p3 = Conv1D(width[2], width[3], 3, s=2)
        self.pan_p4 = C3K2_1D(width[3] * 2, width[3])

        self.down_p4 = Conv1D(width[3], width[4], 3, s=2)
        self.pan_p5 = C3K2_1D(width[4] * 2, width[4])

    def forward(self, p3, p4, p5):
        # ---------- FPN ----------
        p5_up = self.up(self.reduce_p5(p5), p4)
        p4_fpn = self.fpn_p4(torch.cat([p4, p5_up], dim=1))

        p4_up = self.up(self.reduce_p4(p4_fpn), p3)
        p3_fpn = self.fpn_p3(torch.cat([p3, p4_up], dim=1))

        # ---------- PAN ----------
        p3_down = self.down_p3(p3_fpn)
        p3_down = self.up(p3_down, p4_fpn)  # safety alignment
        p4_pan = self.pan_p4(torch.cat([p4_fpn, p3_down], dim=1))

        p4_down = self.down_p4(p4_pan)
        p4_down = self.up(p4_down, p5)      # safety alignment
        p5_pan = self.pan_p5(torch.cat([p5, p4_down], dim=1))

        return p3_fpn, p4_pan, p5_pan

class ForecastHead(nn.Module):
    def __init__(self, channels, horizon):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(sum(channels), horizon)

    def forward(self, feats):
        pooled = [self.pool(f).squeeze(-1) for f in feats]
        return self.fc(torch.cat(pooled, dim=1))

"""
MultiScalingForecastHead creates prediction without having aggregation,
it applies different level aggregation to p3, p4, and p5 and generate the prediction based on concantenation of those aggregations
"""

class MultiScaleForecastHead(nn.Module):
    def __init__(self, channels, horizon):
        super().__init__()

        # Per-scale feature refinement (NO temporal collapse)
        self.p3_head = nn.Sequential(
            Conv1D(channels[0], 64, 3),
            Conv1D(64, 32, 1)
        )

        self.p4_head = nn.Sequential(
            Conv1D(channels[1], 32, 1)
        )

        self.p5_head = nn.Sequential(
            Conv1D(channels[2], 32, 1)
        )

        # Lazy FC infers input dim dynamically
        self.fc = nn.LazyLinear(horizon)

    def forward(self, feats):
        p3, p4, p5 = feats

        f3 = self.p3_head(p3)   # (B, 32, T3)
        f4 = self.p4_head(p4)   # (B, 32, T4)
        f5 = self.p5_head(p5)   # (B, 32, T5)

        # Preserve temporal structure: flatten time
        f3 = f3.flatten(start_dim=1)
        f4 = f4.flatten(start_dim=1)
        f5 = f5.flatten(start_dim=1)

        fused = torch.cat([f3, f4, f5], dim=1)
        return self.fc(fused)

"""
SeparateMultiScaleForecastHead is designed to generate separate output for low frequency, middle frequency and high frequency output

These output will be used for creating short-term, seasonal term and trend-term which is trained with each own loss

Then these output will be finally fused together to create a final prediction.

Thus, this structure depends on multi-objective learning with intentionally creating bias for each component as if it exploits STL concept without explictly using period or timing information.

It also gives opportunity to further optimize the performance by adjusting the contribution of each component.
"""

class SeparateMultiScaleForecastHead(nn.Module):
    def __init__(self, channels, horizon):
        super().__init__()

        # Per-scale feature refinement (NO temporal collapse)
        self.p3_head = nn.Sequential(
            Conv1D(channels[0], 64, 3),
            Conv1D(64, 32, 1)
        )

        self.p4_head = nn.Sequential(
            Conv1D(channels[1], 32, 1)
        )

        self.p5_head = nn.Sequential(
            Conv1D(channels[2], 32, 1)
        )

        # Per-scale fc (NO temporal collapse)
        self.p3_fc = nn.Sequential(
            nn.LazyLinear(horizon),
            nn.SiLU()
        )

        self.p4_fc = nn.Sequential(
            nn.LazyLinear(horizon),
            nn.SiLU()
        )

        self.p5_fc = nn.Sequential(
            nn.LazyLinear(horizon),
            nn.SiLU()
        )

    def forward(self, feats):
        p3, p4, p5 = feats

        f3 = self.p3_head(p3)   # (B, 32, T3)
        f4 = self.p4_head(p4)   # (B, 32, T4)
        f5 = self.p5_head(p5)   # (B, 32, T5)

        # Preserve temporal structure: flatten time
        f3 = f3.flatten(start_dim=1)
        f4 = f4.flatten(start_dim=1)
        f5 = f5.flatten(start_dim=1)

        f3 = self.p3_fc(f3)
        f4 = self.p4_fc(f4)
        f5 = self.p5_fc(f5)

        return f3, f4, f5

class FusionSTL(nn.Module):
    def __init__(self, horizon):
        super().__init__()
        self.fc = nn.Linear(3*horizon, horizon)

    def forward(self, f3, f4, f5):
        return self.fc(torch.cat((f3,f4,f5), dim=1))

class YOLO11_TSF(nn.Module):
    def __init__(
        self,
        horizon=10,
        use_skip=True,
        use_revin=True,
        seq_len=96,
        patch_len=16,
        patch_stride=8,
        return_components=False,
    ):
        super().__init__()
        self.use_skip = use_skip
        self.use_revin = use_revin
        self.return_components = return_components

        if use_revin:
            self.revin = RevIN(num_features=1, affine=True)

        self.patch = PatchTST_Embedding(
            seq_len=seq_len,
            patch_len=patch_len,
            stride=patch_stride,
            d_model=32,
        )

        self.backbone = Backbone(
            width=[32, 64, 128, 256, 256]
        )

        self.neck = Neck(
            width=[32, 64, 128, 256, 256]
        )

        self.head = SeparateMultiScaleForecastHead(
            channels=[128, 256, 256],
            horizon=horizon,
        )

        self.fusion = FusionSTL(
            horizon=horizon
        )

        self.log_vars = nn.Parameter(torch.zeros(4))

        self.log_sigma_p5 = nn.Parameter(torch.zeros(1))

    def forward(self, x, return_components=None):
        if return_components is None:
            return_components = self.return_components

        last_val = x[:, :, -1:] if self.use_skip else 0

        if self.use_revin:
            x = self.revin(x, mode="norm")

        z = self.patch(x)
        p3, p4, p5 = self.backbone(z)
        feats = self.neck(p3, p4, p5)
        hp3, hp4, hp5 = self.head(feats)
        out = self.fusion(hp3, hp4, hp5)

        if self.use_revin:
            mode = "denorm_delta" if self.use_skip else "denorm"
            out = self.revin(out, mode=mode)
            hp3 = self.revin(hp3, mode=mode)
            hp4 = self.revin(hp4, mode=mode)
            hp5 = self.revin(hp5, mode=mode)

        if self.use_skip:
            out = out + last_val.squeeze(1)

        if return_components:
            return out, hp3, hp4, hp5

        return out

if __name__ == "__main__":
    # Model training
    import torch

    def spectral_decompose(x, low_frac=0.05, mid_frac=0.2):
        """
        x: (T,) or (B, T)
        """
        X = torch.fft.rfft(x, dim=-1)
        T = X.shape[-1]

        low = X.clone()
        mid = X.clone()
        high = X.clone()

        low[..., int(low_frac*T):] = 0
        mid[..., :int(low_frac*T)] = 0
        mid[..., int(mid_frac*T):] = 0
        high[..., :int(mid_frac*T)] = 0

        return (
            torch.fft.irfft(low, dim=-1),
            torch.fft.irfft(mid, dim=-1),
            torch.fft.irfft(high, dim=-1),
        )
    # res = spectral_decompose(Train_data)

    def standard_normalziation(x):
        mean_x = x.mean(dim=1)
        std_x = x.std(dim=1)
        y = (x-mean_x)/std_x
        return y, mean_x, std_x

    def inverse_normalization(y,mean_x,std_x):
        return y*std_x + mean_x

    import statsmodels.api as sm
    import numpy as np
    from statsmodels.tsa.seasonal import STL

    data = sm.datasets.co2.load_pandas().data.fillna(method="bfill")
    values = data["co2"].values[:2000]

    batchdata = torch.tensor(values,dtype=torch.float32).view(50,40)
    Train_data = batchdata[:40,:]
    Test_data = batchdata[40:,:]

    #Norm_Train_data, Train_mean, Train_std =  standard_normalziation(Train_data)
    #Norm__data, Train_mean, Train_std =  standard_normalziation(Train_data)


    trend, seasonal, resid  = spectral_decompose(Train_data)

    #lookback horizon of 30, and forecasting horizon of 10
    X = Train_data[:,:30].view(40,1,30)
    y = Train_data[:,30:].view(40,10)
    y_t = trend[:40,30:].view(40,10)
    y_s = seasonal[:40,30:].view(40,10)
    y_r = resid[:40,30:].view(40,10)

    model = YOLO11_TSF(
        horizon=10,
        use_skip=False,
        use_revin=False,
        seq_len=X.shape[-1],
        patch_len=8,
        patch_stride=4,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.001)
    loss_fn = nn.MSELoss()

    def curvature_loss(x,y):
        return torch.mean((x[:, 2:] - 2*y[:, 1:-1] + x[:, :-2])**2)

    def slope_loss(pred, target):
        return F.mse_loss(pred[:, 1:] - pred[:, :-1],
                          target[:, 1:] - target[:, :-1])

    for epoch in range(1000):
        optimizer.zero_grad()
        pred, pred_p3, pred_p4, pred_p5 = model(X, return_components=True)

        #loss = loss_fn(pred, y) + loss_fn(pred_p3, y_r) +  loss_fn(pred_p4, y_s) + slope_loss(pred_p5, y_t)
        #loss = torch.exp(-model.log_vars[0])*loss_fn(pred, y) + torch.exp(-model.log_vars[1])*loss_fn(pred_p3, y_r) +  torch.exp(-model.log_vars[2])*loss_fn(pred_p4, y_s) + torch.exp(-model.log_vars[3])*slope_loss(pred_p5, y_t)
        loss = loss_fn(pred, y) + loss_fn(pred_p3, y_r) +  loss_fn(pred_p4, y_s) + torch.exp(-model.log_sigma_p5)*loss_fn(pred_p5, y_t)# + model.log_sigma_p5

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            loss_weight = [torch.exp(-model.log_vars[k]) for k in range(4)]
            loss_weight = torch.exp(-model.log_sigma_p5)
            print(f"Epoch {epoch}, Loss {loss.item():.4f} Loss Weight {loss_weight}")

    model.eval()
    X_test = Test_data[:,:30].view(10,1,30)
    y_test = Test_data[:,30:].view(10,10)
    y_pred = model(X_test)

    print(f"Prediction: {y_pred}")
    print(f"GroundTruth: {y_test}")

    model.train()
    for epoch in range(500):
        optimizer.zero_grad()
        pred, pred_p3, pred_p4, pred_p5 = model(X, return_components=True)
        loss = loss_fn(pred, y) + loss_fn(pred_p3, y_r) +  loss_fn(pred_p4, y_s) + 0.25*loss_fn(pred_p5, y_t)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss {loss.item():.4f}")

    model.eval()
    X_test = Test_data[:,:30].view(10,1,30)
    y_test = Test_data[:,30:].view(10,10)
    y_pred = model(X_test)
    print(f"Prediction: {y_pred}")
    print(f"GroundTruth: {y_test}")