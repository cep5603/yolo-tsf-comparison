"""
Shared layers for YOLO11-TSF models.

Contains:
- RevIN: Reversible Instance Normalization
- PatchTST_Embedding: PatchTST-style patching with learnable projection and positional encoding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RevIN(nn.Module):
    """
    Reference: https://github.com/ts-kim/RevIN
    """
    
    def __init__(self, num_features=1, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))
        
        # Statistics stored during normalization
        self.mean = None
        self.stdev = None
    
    def forward(self, x, mode):
        """
        Args:
            x: (batch, channels, seq_len) for 'norm', (batch, horizon) for 'denorm'
            mode: 'norm' or 'denorm'
        """
        if mode == 'norm':
            return self._normalize(x)
        elif mode == 'denorm':
            return self._denormalize(x)
        elif mode == 'denorm_delta':
            return self._denormalize_delta(x)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def _normalize(self, x):
        # x: (batch, channels, seq_len)
        # Compute stats over temporal dimension
        self.mean = x.mean(dim=-1, keepdim=True).detach()  # (batch, channels, 1)
        self.stdev = torch.sqrt(x.var(dim=-1, keepdim=True, unbiased=False) + self.eps).detach()
        
        x = (x - self.mean) / self.stdev
        
        if self.affine:
            # affine_weight/bias: (num_features,) -> broadcast to (1, num_features, 1)
            x = x * self.affine_weight.view(1, -1, 1)
            x = x + self.affine_bias.view(1, -1, 1)
        
        return x
    
    def _denormalize(self, x):
        # x: (batch, horizon) or (batch, channels, horizon)
        # Expand dims if needed
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, horizon)
        
        if self.affine:
            x = x - self.affine_bias.view(1, -1, 1)
            x = x / (self.affine_weight.view(1, -1, 1) + self.eps)
        
        # mean/stdev: (batch, channels, 1)
        x = x * self.stdev
        x = x + self.mean
        
        return x.squeeze(1)  # (batch, horizon)

    def _denormalize_delta(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        if self.affine:
            x = x - self.affine_bias.view(1, -1, 1)
            x = x / (self.affine_weight.view(1, -1, 1) + self.eps)

        x = x * self.stdev
        return x.squeeze(1)


class PatchTST_Embedding(nn.Module):
    """
    1. Pad sequence end w/ replicated last value
    2. Unfold into overlapping patches
    3. Linear projection to embedding dimension
    4. Add learnable positional encoding
    """
    
    def __init__(self, seq_len, patch_len=16, stride=8, d_model=32, padding_patch=True):
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.padding_patch = padding_patch
        
        # Calculate # of patches
        if padding_patch:
            # Pad sequence to ensure at least 1 full patch at end
            self.pad_len = stride
            padded_len = seq_len + self.pad_len
        else:
            self.pad_len = 0
            padded_len = seq_len
        
        self.n_patches = (padded_len - patch_len) // stride + 1
        
        # Linear projection: patch_len -> d_model
        self.patch_proj = nn.Linear(patch_len, d_model)
        
        # Learnable positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x):
        """
        Args:
            x: (batch, channels, seq_len)
        
        Returns:
            (batch, d_model, n_patches)
        """
        batch, channels, seq_len = x.shape
        
        # Pad end with last value
        if self.padding_patch:
            # Replicate last value
            last_val = x[:, :, -1:].expand(-1, -1, self.pad_len)
            x = torch.cat([x, last_val], dim=-1)  # (batch, channels, seq_len + pad_len)
        
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # Extract sliding patches: unfold(dim, size, step) -> (batch, channels, n_patches, patch_len)
        patches = patches.reshape(-1, self.n_patches, self.patch_len)  # Reshape: (batch * channels, n_patches, patch_len)
        patches = self.patch_proj(patches)  # Project: (batch * channels, n_patches, d_model)
        patches = patches + self.pos_embed  # Add positional encoding
        patches = patches.reshape(batch, channels, self.n_patches, self.d_model)  # Reshape back: (batch, channels, n_patches, d_model)
        
        # For single channel, squeeze and transpose: (batch, d_model, n_patches)
        # Note that I have this different from the official PatchTST implementation in PatchTST_supervised/layers/PatchTST_backbone.py:
        # For multi-channel inputs we merge vars into the embedding dims instead of keeping a separate C (but this isn't used anyway)
        if channels == 1:
            patches = patches.squeeze(1).transpose(1, 2)
        else:
            # Multi-channel: (batch, channels * d_model, n_patches)
            patches = patches.permute(0, 1, 3, 2).reshape(batch, channels * self.d_model, self.n_patches)
        
        return patches
