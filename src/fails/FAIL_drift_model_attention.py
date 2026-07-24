"""Attention-Based Drift for differentiable drift estimation.

Predicts drift as attention-weighted combination of time bins:
D(t) = Σ_s α(t,s) · f(s)

where α(t,s) = softmax(-|t-s|/σ + g(z_brain[t], z_brain[s]))

Differentiable by design, naturally smooth, can model complex drift patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from FAIL_drift_model import UnifiedDriftLocalizer


class AttentionDriftLocalizer(UnifiedDriftLocalizer):
    """UnifiedDriftLocalizer with attention-based drift prediction.
    
    Instead of learning D(t) directly, learn attention weights over time bins.
    This is differentiable by construction and naturally enforces smoothness.
    """
    
    def __init__(self, *args, attn_heads=4, attn_temporal_sigma=100.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.attn_heads = attn_heads
        self.attn_temporal_sigma = attn_temporal_sigma
        
        # Replace global_drift with attention mechanism
        # We'll still store a base drift, but it's modulated by attention
        del self.global_drift
        
        # Attention parameters
        self.attn_query = nn.Parameter(torch.randn(attn_heads, self.num_bins) * 0.01)
        self.attn_key = nn.Parameter(torch.randn(attn_heads, self.num_bins) * 0.01)
        self.attn_value = nn.Parameter(torch.zeros(attn_heads, self.num_bins))
        
        # Base drift (learned residual)
        self.base_drift = nn.Parameter(torch.zeros(self.num_bins))
    
    def get_attention_drift(self):
        """Compute drift as attention-weighted combination."""
        device = self.base_drift.device
        
        # Create temporal position embeddings
        t = torch.arange(self.num_bins, device=device, dtype=torch.float32)
        
        # Multi-head attention
        drift_components = []
        for h in range(self.attn_heads):
            # Temporal distance matrix
            t_diff = t.unsqueeze(1) - t.unsqueeze(0)  # (T, T)
            temporal_bias = -torch.abs(t_diff) / (self.attn_temporal_sigma / self.bin_width_sec)
            
            # Attention scores
            scores = self.attn_query[h].unsqueeze(1) + self.attn_key[h].unsqueeze(0) + temporal_bias
            attn_weights = F.softmax(scores, dim=1)
            
            # Weighted sum
            drift_h = torch.sum(attn_weights * self.attn_value[h].unsqueeze(0), dim=1)
            drift_components.append(drift_h)
        
        # Combine heads and add base drift
        drift = torch.stack(drift_components, dim=0).mean(dim=0) + self.base_drift
        
        # Gauge pin
        drift = drift - drift[0]
        return drift
    
    def get_drift(self, times_sec):
        """Sample D(t) at given times using attention-based drift."""
        drift_full = self.get_attention_drift()
        
        bin_coords = times_sec / self.bin_width_sec
        normalized = (bin_coords / max(self.num_bins - 1, 1)) * 2.0 - 1.0
        normalized = normalized.clamp(-1.0, 1.0)
        
        grid_input = torch.stack([normalized, torch.zeros_like(normalized)], dim=-1)
        grid_input = grid_input.unsqueeze(0).unsqueeze(0)
        
        drift_grid = drift_full.view(1, 1, 1, -1)
        sampled = F.grid_sample(drift_grid, grid_input, align_corners=True,
                                mode='bilinear', padding_mode='border')
        return sampled.squeeze(0).squeeze(0).squeeze(0)
    
    def freeze_drift(self):
        """Freeze attention parameters."""
        self.attn_query.requires_grad = False
        self.attn_key.requires_grad = False
        self.attn_value.requires_grad = False
        self.base_drift.requires_grad = False
    
    def unfreeze_all(self):
        """Unfreeze all parameters."""
        super().unfreeze_all()
        self.attn_query.requires_grad = True
        self.attn_key.requires_grad = True
        self.attn_value.requires_grad = True
        self.base_drift.requires_grad = True
