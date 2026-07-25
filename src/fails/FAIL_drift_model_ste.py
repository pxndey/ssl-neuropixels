"""Straight-Through Estimator (STE) for differentiable drift estimation.

Uses hard argmax in forward pass (exact DREDge-style) but soft gradient in backward.
This gives the encoder the "correct" drift signal while still allowing gradients to flow.
"""

import torch
import torch.nn.functional as F

from FAIL_drift_model import UnifiedDriftLocalizer


class STEDriftLocalizer(UnifiedDriftLocalizer):
    """UnifiedDriftLocalizer with Straight-Through Estimator for dredge loss.
    
    Forward: hard argmax (exact shift)
    Backward: soft argmax (gradient flow)
    """
    
    def __init__(self, *args, ste_beta_forward=1000.0, ste_beta_backward=3.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.ste_beta_forward = ste_beta_forward
        self.ste_beta_backward = ste_beta_backward
    
    def compute_diff_dredge_loss_ste(self, R):
        """STE version: hard forward, soft backward."""
        T, grid_len = R.size()
        if T < 2:
            return torch.tensor(0.0, device=R.device)
        
        # Normalize
        row_norm = R.norm(dim=1, keepdim=True).clamp(min=1e-8)
        R_norm = R / row_norm
        
        max_shift = self.max_shift_bins
        padded = F.pad(R_norm, (max_shift, max_shift), mode='constant', value=0)
        shifts = torch.arange(-max_shift, max_shift + 1, dtype=torch.float32, device=R.device)
        
        loss = torch.tensor(0.0, device=R.device)
        denom = 0.0
        
        for w in range(1, self.window_bins + 1):
            if T <= w:
                break
            n = T - w
            signals = padded[0:n].unsqueeze(0)
            kernels = R_norm[w:T].unsqueeze(1)
            corr = F.conv1d(signals, kernels, groups=n).squeeze(0)
            
            # Hard argmax for forward (detach to stop gradient)
            hard_probs = F.softmax(corr * self.ste_beta_forward, dim=-1)
            hard_est = torch.sum(shifts.unsqueeze(0) * hard_probs.detach(), dim=-1)
            
            # Soft argmax for backward only
            soft_probs = F.softmax(corr * self.ste_beta_backward, dim=-1)
            soft_est = torch.sum(shifts.unsqueeze(0) * soft_probs, dim=-1)
            
            # STE: forward uses hard, backward uses soft
            est = hard_est - soft_est.detach() + soft_est
            
            decay = torch.exp(torch.tensor(-float(w) / self.window_bins, device=R.device))
            loss = loss + decay * torch.mean(est ** 2)
            denom += 1.0
        
        return loss / max(denom, 1.0)
    
    def forward(self, wf, coords, mask, centroid, times_sec, gt_ptp=None, phase="all"):
        """Override to use STE dredge loss."""
        x, y, z_local, alpha, xc, zc = self._encoder_forward(wf, coords, mask)
        drift = self.get_drift(times_sec)
        
        loss_monopole = torch.tensor(0.0, device=wf.device)
        if phase in ("monopole", "all"):
            dz = self.monopole_decoder(z_local, drift, zc)
            dx = x.unsqueeze(-1) - xc
            r2 = dx ** 2 + dz ** 2 + y.unsqueeze(-1) ** 2 + self.channel_insulation_constant ** 2
            ptp_pred = alpha.unsqueeze(-1) / torch.sqrt(r2)
            if gt_ptp is None:
                from model import compute_feature
                gt_ptp = compute_feature(wf, self.recon_feature)
            from model import masked_recon_loss
            loss_monopole = masked_recon_loss(gt_ptp, ptp_pred, mask, self.loss_type)
        
        loss_dredge = torch.tensor(0.0, device=wf.device)
        loss_smooth = torch.tensor(0.0, device=wf.device)
        if phase in ("dredge", "all"):
            z_probe = centroid[:, 1] + z_local
            z_brain_driftcorr = z_probe - drift
            R = self.soft_rasterize(z_brain_driftcorr, alpha, times_sec)
            loss_dredge = self.compute_diff_dredge_loss_ste(R)  # Use STE!
            loss_smooth = torch.mean(torch.diff(self.global_drift) ** 2)
        
        extras = {
            "x_brain_abs": centroid[:, 0] + x,
            "z_brain_abs": centroid[:, 1] + z_local,
            "z_brain_driftcorr": centroid[:, 1] + z_local - drift,
            "y": y, "alpha": alpha,
            "drift": drift,
            "ptp_pred": ptp_pred if phase in ("monopole", "all") else None,
        }
        return loss_monopole, loss_dredge, loss_smooth, extras
