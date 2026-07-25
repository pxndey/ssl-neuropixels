"""Direct Temporal Coherence Loss for differentiable drift estimation.

Instead of cross-correlation + soft-argmax, directly penalize temporal variance of z_brain.
This is differentiable by construction and directly optimizes what we want:
same neuron → same z_brain at different times.
"""

import torch
import torch.nn.functional as F

from FAIL_drift_model import UnifiedDriftLocalizer


class CoherenceDriftLocalizer(UnifiedDriftLocalizer):
    """UnifiedDriftLocalizer with direct temporal coherence loss.
    
    Loss: L = Σ_{i,j} |z_brain[i] - z_brain[j]|² · w(t_i, t_j) · sim(i,j)
    where w(t_i, t_j) = exp(-|t_i - t_j|/τ) and sim(i,j) based on amplitude similarity
    """
    
    def __init__(self, *args, temporal_window_sec=300.0, coherence_temperature=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.temporal_window_sec = temporal_window_sec
        self.coherence_temperature = coherence_temperature
    
    def compute_temporal_coherence_loss(self, z_brain, alpha, times_sec):
        """Direct temporal coherence: penalize variance of z_brain over time.
        
        For efficiency, we compute this per time bin rather than all pairs.
        """
        # Group spikes by time bin
        bin_ids = torch.clamp((times_sec / self.bin_width_sec).long(), 
                              0, self.num_bins - 1)
        
        # Compute mean z_brain per time bin (weighted by amplitude)
        z_weighted = z_brain * alpha
        z_mean = torch.zeros(self.num_bins, device=z_brain.device)
        z_weight_sum = torch.zeros(self.num_bins, device=z_brain.device)
        
        z_mean.index_add_(0, bin_ids, z_weighted)
        z_weight_sum.index_add_(0, bin_ids, alpha)
        
        # Avoid division by zero
        mask = z_weight_sum > 0
        z_mean[mask] = z_mean[mask] / z_weight_sum[mask]
        
        # Compute temporal coherence: penalize difference between nearby time bins
        loss = torch.tensor(0.0, device=z_brain.device)
        window_bins = int(self.temporal_window_sec / self.bin_width_sec)
        
        for w in range(1, min(window_bins + 1, self.num_bins)):
            valid = mask[:-w] & mask[w:]
            if valid.any():
                diff = (z_mean[:-w][valid] - z_mean[w:][valid]) ** 2
                weight = torch.exp(-torch.tensor(w / window_bins, device=z_brain.device))
                loss = loss + weight * diff.mean()
        
        return loss / max(window_bins, 1)
    
    def forward(self, wf, coords, mask, centroid, times_sec, gt_ptp=None, phase="all"):
        """Override to use temporal coherence loss instead of dredge."""
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
        
        loss_coherence = torch.tensor(0.0, device=wf.device)
        loss_smooth = torch.tensor(0.0, device=wf.device)
        if phase in ("dredge", "all"):
            z_probe = centroid[:, 1] + z_local
            z_brain_driftcorr = z_probe - drift
            # Use temporal coherence instead of cross-correlation!
            loss_coherence = self.compute_temporal_coherence_loss(
                z_brain_driftcorr, alpha, times_sec)
            loss_smooth = torch.mean(torch.diff(self.global_drift) ** 2)
        
        extras = {
            "x_brain_abs": centroid[:, 0] + x,
            "z_brain_abs": centroid[:, 1] + z_local,
            "z_brain_driftcorr": centroid[:, 1] + z_local - drift,
            "y": y, "alpha": alpha,
            "drift": drift,
            "ptp_pred": ptp_pred if phase in ("monopole", "all") else None,
        }
        return loss_monopole, loss_coherence, loss_smooth, extras
