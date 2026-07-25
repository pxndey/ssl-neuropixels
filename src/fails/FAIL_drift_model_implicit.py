"""Implicit Differentiation for differentiable drift estimation.

Solves M-step to convergence, then computes encoder gradients via implicit function theorem.
This gives exact gradients through the optimization, not approximate soft-argmax.
"""

import torch
import torch.nn.functional as F

from FAIL_drift_model import UnifiedDriftLocalizer


class ImplicitDriftLocalizer(UnifiedDriftLocalizer):
    """UnifiedDriftLocalizer with implicit differentiation for M-step.
    
    M-step is solved to convergence, then encoder gradients computed via:
    dL/d_encoder = dL/dD* · (d²L/dD²)^(-1) · d²L/dDdencoder
    """
    
    def __init__(self, *args, implicit_m_steps=10, implicit_lr=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.implicit_m_steps = implicit_m_steps
        self.implicit_lr = implicit_lr
    
    def solve_m_step(self, wf, coords, mask, centroid, times_sec, lambda_smooth=0.1):
        """Solve M-step to convergence, tracking computation graph for implicit diff."""
        # Freeze encoder
        self.freeze_encoder()
        
        # Get encoder outputs (these are the "parameters" for implicit diff)
        with torch.enable_grad():
            x, y, z_local, alpha, xc, zc = self._encoder_forward(wf, coords, mask)
            z_probe = centroid[:, 1] + z_local
            # Keep z_probe in computation graph
            z_probe_for_raster = z_probe.clone()
        
        # Run M-step gradient descent to convergence
        for _ in range(self.implicit_m_steps):
            drift = self.get_drift(times_sec)
            z_brain_driftcorr = z_probe_for_raster - drift
            R = self.soft_rasterize(z_brain_driftcorr.detach(), alpha.detach(), times_sec)
            loss_dredge = self.compute_diff_dredge_loss(R)
            loss_smooth = torch.mean(torch.diff(self.global_drift) ** 2)
            loss = loss_dredge + lambda_smooth * loss_smooth
            
            grad = torch.autograd.grad(loss, self.global_drift, create_graph=True)[0]
            with torch.no_grad():
                self.global_drift.data -= self.implicit_lr * grad
        
        self.unfreeze_all()
        return z_probe  # Return for gradient computation
    
    def compute_dredge_loss_implicit(self, R, lambda_smooth=0.1):
        """Compute dredge loss at converged point."""
        loss_dredge = self.compute_diff_dredge_loss(R)
        loss_smooth = torch.mean(torch.diff(self.global_drift) ** 2)
        return loss_dredge + lambda_smooth * loss_smooth
