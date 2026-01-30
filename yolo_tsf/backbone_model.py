import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import RevIN, PatchTST_Embedding

class Conv1D(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None):
        super().__init__()
        self.conv = nn.Conv1d(c1, c2, k, s, padding=k//2 if p is None else p, bias=False)
        self.bn = nn.BatchNorm1d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

"""
C3K2 block (YOLO11-style, 1D)

Kernel mixing (1×1 + 3×3)

CSP split

Lightweight bottlenecks
"""

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

# SPPF (1D version, YOLO11-compatible) : Spatial Pyramid Pooling Fast

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

"""
C2PSA block (YOLO11-style attention, 1D) : Partial Spatial Attention

Channel Attention learns “which channels/features are important?”
Spatial Attention learns “which temporal positions are important?”
"""

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

# Patch Embedding (overlapping temporal patches)

class PatchEmbed1D(nn.Module):
    def __init__(self, in_ch, embed_dim, patch_len, stride):
        super().__init__()
        self.proj = nn.Conv1d(in_ch, embed_dim, patch_len, stride)

    def forward(self, x):
        return self.proj(x)

# YOLO11-style backbone

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

# Forecasting Head (multi-scale aggregation)

class ForecastHead(nn.Module):
    def __init__(self, channels, horizon):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(sum(channels), horizon)

    def forward(self, feats):
        pooled = [self.pool(f).squeeze(-1) for f in feats]
        return self.fc(torch.cat(pooled, dim=1))

# Full YOLO11-inspired TS model with skip connection

class YOLO11_TSF(nn.Module):
    def __init__(self, horizon=10, use_skip=True, use_revin=True, seq_len=96, patch_len=16, patch_stride=8):
        super().__init__()
        self.horizon = horizon
        self.use_skip = use_skip
        self.use_revin = use_revin
        
        # RevIN for instance normalization
        if use_revin:
            self.revin = RevIN(num_features=1, affine=True)
        
        # PatchTST-style embedding with positional encoding
        self.patch = PatchTST_Embedding(
            seq_len=seq_len,
            patch_len=patch_len,
            stride=patch_stride,
            d_model=32
        )

        self.backbone = Backbone(
            width=[32, 64, 128, 256, 256]
        )

        self.head = ForecastHead(
            channels=[128, 256, 256],
            horizon=horizon
        )

    def forward(self, x):
        # x: (batch, 1, seq_len)
        last_val = x[:, :, -1] if self.use_skip else 0
        
        # RevIN normalize
        if self.use_revin:
            x = self.revin(x, mode='norm')
        
        z = self.patch(x)
        feats = self.backbone(z)
        out = self.head(feats)  # (batch, horizon)
        
        # RevIN denormalize
        if self.use_revin:
            out = self.revin(out, mode='denorm')
        
        # Skip connection (adds last value AFTER denorm)
        if self.use_skip:
            out = out + last_val.squeeze(1)
        
        return out