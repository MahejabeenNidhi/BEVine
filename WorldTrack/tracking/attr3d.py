# WorldTrack/tracking/attr3d.py
"""Per-track 3D attribute (yaw / size / posture) state estimation.
"""

from collections import deque

import numpy as np


def wrap_pi(a):
    """Wrap an angle to [-pi, pi)."""
    return (float(a) + np.pi) % (2.0 * np.pi) - np.pi


class Attr3DState:
    VALID_SIZE_MODES = ('mean', 'ema', 'median')
    VALID_YAW_MODES = ('ema', 'kf')

    def __init__(self,
                 size_mode='mean', size_alpha=0.3, size_reject_cm=0.0,
                 median_window=15,
                 yaw_mode='ema', yaw_alpha=0.3, flip_align=True,
                 dt=1.0, yaw_q=4e-2, yaw_r=0.25,
                 posture_alpha=0.3, posture_hysteresis=0.1):
        if size_mode not in self.VALID_SIZE_MODES:
            raise ValueError(f"size_mode must be one of "
                             f"{self.VALID_SIZE_MODES}, got {size_mode!r}")
        if yaw_mode not in self.VALID_YAW_MODES:
            raise ValueError(f"yaw_mode must be one of "
                             f"{self.VALID_YAW_MODES}, got {yaw_mode!r}")

        self.size_mode = size_mode
        self.size_alpha = float(size_alpha)
        self.size_reject_cm = float(size_reject_cm)
        self.yaw_mode = yaw_mode
        self.yaw_alpha = float(yaw_alpha)
        self.flip_align = bool(flip_align)
        self.dt = float(dt)
        self.posture_alpha = float(posture_alpha)
        self.posture_hysteresis = float(posture_hysteresis)

        # ── size state ──
        self._size = None                        # (3,) cm
        self._size_sum = np.zeros(3, dtype=np.float64)
        self._size_n = 0
        self._size_hist = deque(maxlen=int(median_window))

        # ── yaw state ──
        self._yaw_vec = None                     # unit (sin, cos)
        self._theta = None                       # KF mean (rad)
        self._omega = 0.0                        # KF yaw rate (rad/frame)
        self._P = np.diag([0.5, 0.5]).astype(np.float64)
        self._Q = np.diag([float(yaw_q), float(yaw_q)]).astype(np.float64)
        self._R = float(yaw_r)

        # ── posture state ──
        self._post_p = None
        self._post_cls = 0

        self.n_obs = 0
        self.n_flips = 0
        self.flipped_last = False

    # ------------------------------------------------------------------
    # motion model (called once per frame, before association)
    # ------------------------------------------------------------------
    def predict(self):
        """Advance the yaw KF by one frame. No-op in 'ema' mode
        (a constant-heading model needs no propagation)."""
        if self.yaw_mode != 'kf' or self._theta is None:
            return
        F = np.array([[1.0, self.dt], [0.0, 1.0]], dtype=np.float64)
        self._theta = wrap_pi(self._theta + self._omega * self.dt)
        self._P = F @ self._P @ F.T + self._Q
        self._yaw_vec = np.array([np.sin(self._theta),
                                  np.cos(self._theta)], dtype=np.float64)

    # ------------------------------------------------------------------
    def update(self, yaw=None, dims=None, posture_prob=None):
        """Fuse one per-frame measurement. Any field may be None."""
        self.flipped_last = False
        touched = False

        if dims is not None:
            d = np.asarray(dims, dtype=np.float64).reshape(-1)[:3]
            if d.size == 3 and np.all(np.isfinite(d)) and np.all(d > 0):
                self._update_size(d)
                touched = True

        if yaw is not None and np.isfinite(float(yaw)):
            self._update_yaw(float(yaw))
            touched = True

        if posture_prob is not None and np.isfinite(float(posture_prob)):
            p = float(posture_prob)
            a = self.posture_alpha
            self._post_p = p if self._post_p is None \
                else (1.0 - a) * self._post_p + a * p
            hi = 0.5 + self.posture_hysteresis
            lo = 0.5 - self.posture_hysteresis
            if self._post_p >= hi:
                self._post_cls = 1
            elif self._post_p <= lo:
                self._post_cls = 0
            # inside the dead band -> KEEP the previous class (hysteresis)
            touched = True

        if touched:
            self.n_obs += 1

    # ------------------------------------------------------------------
    def _update_size(self, d):
        # Reject a wild single-frame reading once we have a stable estimate
        # (a size head sampled on a partially occluded animal can be 2x off).
        if (self.size_reject_cm > 0.0 and self._size is not None
                and self.n_obs >= 3
                and float(np.max(np.abs(d - self._size))) > self.size_reject_cm):
            return

        if self.size_mode == 'ema':
            a = self.size_alpha
            self._size = d.copy() if self._size is None \
                else (1.0 - a) * self._size + a * d
        elif self.size_mode == 'mean':
            self._size_sum += d
            self._size_n += 1
            self._size = self._size_sum / float(self._size_n)
        else:  # 'median'
            self._size_hist.append(d.copy())
            self._size = np.median(np.stack(self._size_hist, axis=0), axis=0)

    # ------------------------------------------------------------------
    def _update_yaw(self, y):
        if self.yaw_mode == 'ema':
            v = np.array([np.sin(y), np.cos(y)], dtype=np.float64)
            if self._yaw_vec is None:
                self._yaw_vec = v
                return
            if self.flip_align and float(v @ self._yaw_vec) < 0.0:
                v = -v
                self.n_flips += 1
                self.flipped_last = True
            a = self.yaw_alpha
            nv = (1.0 - a) * self._yaw_vec + a * v
            n = float(np.linalg.norm(nv))
            self._yaw_vec = (nv / n) if n > 1e-6 else v
            return

        # ── 'kf': 2-state (theta, omega), scalar angle measurement ──
        z = wrap_pi(y)
        if self._theta is None:
            self._theta = z
            self._omega = 0.0
            self._P = np.diag([self._R, 1.0]).astype(np.float64)
        else:
            if self.flip_align and abs(wrap_pi(z - self._theta)) > (np.pi / 2.0):
                z = wrap_pi(z + np.pi)
                self.n_flips += 1
                self.flipped_last = True
            r = wrap_pi(z - self._theta)          # wrapped innovation
            S = self._P[0, 0] + self._R
            K = self._P[:, 0] / max(S, 1e-9)      # (2,)
            self._theta = wrap_pi(self._theta + K[0] * r)
            self._omega = float(self._omega + K[1] * r)
            self._P = self._P - np.outer(K, self._P[0, :])
        self._yaw_vec = np.array([np.sin(self._theta),
                                  np.cos(self._theta)], dtype=np.float64)

    # ------------------------------------------------------------------
    @property
    def has_size(self):
        return self._size is not None

    @property
    def has_yaw(self):
        return self._yaw_vec is not None

    @property
    def dims(self):
        """(length, width, height) in CENTIMETRES, or None."""
        return None if self._size is None else self._size.copy()

    @property
    def yaw(self):
        """Smoothed heading in radians, same convention as
        decode.decoder(): atan2(sin, cos)."""
        if self._yaw_vec is None:
            return None
        return float(np.arctan2(self._yaw_vec[0], self._yaw_vec[1]))

    @property
    def yaw_rate(self):
        return float(self._omega)

    @property
    def posture_prob(self):
        return None if self._post_p is None else float(self._post_p)

    @property
    def posture_class(self):
        return int(self._post_cls)
