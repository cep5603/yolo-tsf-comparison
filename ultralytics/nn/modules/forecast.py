import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Sequence


class Forecast(nn.Module):
    """
    Multi-scale forecasting head for 1D time series represented as (B, C, 1, L).

    Expects a list of neck features: [(B, Ci, Hi, Wi), ...].
    Height is averaged to 1, all scales are upsampled to the max width (time),
    then fused and read out at the last time index.

    Args (YAML -> __init__):
      horizon: int, forecast horizon H
      quantiles: Optional[List[float]] for probabilistic forecasts (e.g., [0.1, 0.5, 0.9])
      hidden: int, projection channels per scale
      dropout: float
      agg: "concat" or "sum"
      pool_size: int, size to pool time dimension to
      ch: List[int] injected by parse_model with per-scale channels (appended last)
    """

    export = False  # compatibility with other heads

    def __init__(
        self,
        horizon: int,
        quantiles: Optional[Sequence[float]] = None,
        hidden: int = 256,
        dropout: float = 0.0,
        agg: str = "concat",
        pool_size: int = 1,
        ch: Sequence[int] = (),  # ch must be last to match parse_model behavior
    ):
        super().__init__()
        if isinstance(ch, int):
            ch = [ch]
        assert isinstance(ch, (list, tuple)) and len(ch) >= 1, "Forecast requires list of input channels per scale"
        assert agg in ("concat", "sum")

        self.horizon = int(horizon)
        self.quantiles = list(quantiles) if quantiles is not None else None
        self.nq = 1 if self.quantiles is None else len(self.quantiles)
        self.agg = agg
        self.pool_size = pool_size

        # Per-scale 1x1 projections to hidden channels
        self.proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(c, hidden, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden),
                    nn.SiLU(),
                    nn.Dropout2d(dropout),
                )
                for c in ch
            ]
        )
        
        # If agg=concat, we concat features from all scales. 
        # Each scale contributes (hidden * pool_size) features.
        in_c = hidden * len(ch) * self.pool_size if agg == "concat" else hidden * self.pool_size

        # For pooled-head variant: total feature dim after pooling = in_c
        self.in_c = in_c
        # Linear head mapping concatenated per-scale pooled features -> horizon * nq
        self.fc = nn.Linear(in_c, self.horizon * self.nq)

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        # Forward pass global pooling -> concat -> linear projection
        assert isinstance(feats, (list, tuple)) and len(feats) > 0 # feats is BCHW

        pooled = []
        for f, p in zip(feats, self.proj):
            # project per-scale features to 'hidden' channels
            z = p(f)  # (B, hidden, H, W)
            
            # Adaptive Average Pooling over time dimension (W)
            # We want (B, hidden, 1, pool_size)
            z = F.adaptive_avg_pool2d(z, (1, self.pool_size))
            
            z = z.view(z.size(0), -1)  # (B, hidden * pool_size)
            pooled.append(z)

        if self.agg == "concat":
            x = torch.cat(pooled, dim=1)  # (B, in_c)
        else:
            x = torch.stack(pooled, dim=0).sum(0)  # (B, hidden * pool_size)

        y = self.fc(x)  # (B, H*nq)
        y = y.view(y.size(0), self.horizon, self.nq)  # (B, H, nq)
        return y.squeeze(-1) if self.nq == 1 else y