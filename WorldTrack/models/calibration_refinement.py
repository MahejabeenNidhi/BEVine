# WorldTrack/models/calibration_refinement.py
"""
Learns a single shared 6-DOF extrinsic correction per camera.
All sequences share the same correction because they use the same physical rig.
"""

import numpy as np
import torch
import torch.nn as nn
from utils.geom import so3_exp
from utils.precision import fp32_so3_exp

class CalibrationRefinementModule(nn.Module):
    """
    Learns a single shared 6-DOF correction ΔE_c for each of the
    S cameras.  All sequences use the same correction because they share
    the same physical camera rig.

    Parameterisation
    ----------------
    delta_r : nn.Parameter  shape (S, 3)   axis-angle in so(3)
    delta_t : nn.Parameter  shape (S, 3)   translation offset in world units

    The corrected extrinsic for camera c is assembled as:

        R_refined = so3_exp(delta_r[c]) @ R_base        # left-multiply
        t_refined = t_base + delta_t[c]
        E_refined = [ R_refined | t_refined ]            # (3, 4)

    All parameters are initialised to zero so the module is an identity
    at the start of training and deviates only as the reprojection loss
    pulls it toward a better calibration.
    """

    def __init__(self, num_cameras: int = 4):
        super().__init__()
        self.num_cameras = num_cameras
        self.delta_r = nn.Parameter(torch.zeros(num_cameras, 3))
        self.delta_t = nn.Parameter(torch.zeros(num_cameras, 3))

    def forward(self, base_extrinsics):
        """
        Apply the learned correction to base extrinsics.

        Parameters
        ----------
        base_extrinsics : (B, S, 4, 4) float32, on any device

        Returns
        -------
        refined : (B, S, 4, 4) float32, same device
        """
        B, S = base_extrinsics.shape[:2]
        device = base_extrinsics.device
        assert S == self.num_cameras, (
            f"Expected {self.num_cameras} cameras, got {S}"
        )

        # Rotation corrections: (S, 3, 3)
        delta_R = fp32_so3_exp(self.delta_r).to(device)
        delta_t = self.delta_t.to(device)

        R_base = base_extrinsics[:, :, :3, :3]   # (B, S, 3, 3)
        t_base = base_extrinsics[:, :, :3, 3]    # (B, S, 3)

        # Left-multiply rotation correction
        R_refined = torch.matmul(
            delta_R.unsqueeze(0), R_base
        )  # (B, S, 3, 3)

        # Add translation correction
        t_refined = t_base + delta_t.unsqueeze(0)  # (B, S, 3)

        # Assemble 4×4 without in-place ops (autograd safe)
        top_rows = torch.cat(
            [R_refined, t_refined.unsqueeze(-1)], dim=-1
        )  # (B, S, 3, 4)
        bottom_row = base_extrinsics[:, :, 3:4, :]  # (B, S, 1, 4)
        refined = torch.cat([top_rows, bottom_row], dim=-2)  # (B, S, 4, 4)

        return refined

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def get_delta_norms(self):
        """
        Returns {cam_index: (delta_r_norm, delta_t_norm)} as plain
        Python floats (detached scalars).
        """
        result = {}
        for c in range(self.num_cameras):
            dr = self.delta_r[c].detach().norm().item()
            dt = self.delta_t[c].detach().norm().item()
            result[c] = (dr, dt)
        return result

    def export_refined_extrinsics(self, base_extrinsics_np, cam_names):
        """
        Compose refined 3×4 extrinsics as numpy arrays.

        Parameters
        ----------
        base_extrinsics_np : dict {cam_name: ndarray (3,4)}
        cam_names : list of str, same order as camera indices 0..S-1

        Returns
        -------
        dict {cam_name: ndarray (3,4)} of fully composed refined matrices
        """
        result = {}
        with torch.no_grad():
            delta_R = so3_exp(self.delta_r.cpu())  # (S, 3, 3)

            for c, name in enumerate(cam_names):
                base_rt = base_extrinsics_np[name]     # (3, 4)
                R_base = base_rt[:, :3]                 # (3, 3)
                t_base = base_rt[:, 3]                  # (3,)

                dR = delta_R[c].numpy()                 # (3, 3)
                dt = self.delta_t[c].cpu().numpy()      # (3,)

                R_refined = dR @ R_base
                t_refined = t_base + dt

                Rt = np.hstack([R_refined, t_refined.reshape(3, 1)])
                result[name] = Rt.astype(np.float32)

        return result