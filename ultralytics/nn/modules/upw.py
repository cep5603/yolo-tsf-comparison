import torch
import torch.nn as nn
import torch.nn.functional as F

# This is for width-only upsampling to keep constant height=1 when concatenating
# (see yolo11-forecast.yaml for usage)

class UpW(nn.Module):
    """
    Upsample width (time) only by a factor, keep height unchanged.
    NCHW -> NCHW with H same, W *= scale.
    """

    def __init__(self, scale: float = 2.0, mode: str = "nearest"):
        super().__init__()
        self.scale = float(scale)
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=(1.0, self.scale), mode=self.mode)