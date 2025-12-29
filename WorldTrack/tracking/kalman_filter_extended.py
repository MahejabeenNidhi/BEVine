# WorldTrack/tracking/kalman_filter_extended.py

import numpy as np
import scipy.linalg
from enum import Enum


class MotionModel(Enum):
    CONSTANT_VELOCITY = 1
    CONSTANT_ACCELERATION = 2
    CURVILINEAR = 3
    ADAPTIVE = 4


chi2inv95 = {
    1: 3.8415, 2: 5.9915, 3: 7.8147, 4: 9.4877,
    5: 11.070, 6: 12.592, 7: 14.067, 8: 15.507, 9: 16.919
}


class ExtendedKalmanFilter(object):
    """
    Extended Kalman Filter with multiple motion models for cattle tracking.

    State space options:
    - CONSTANT_VELOCITY: [x, y, vx, vy]
    - CONSTANT_ACCELERATION: [x, y, vx, vy, ax, ay]
    - CURVILINEAR: [x, y, vx, vy, omega] where omega is angular velocity
    - ADAPTIVE: Switches between models based on motion
    """

    def __init__(self, motion_model=MotionModel.ADAPTIVE, dt=1.0):
        self.motion_model = motion_model
        self.dt = dt

        # Choose state dimension based on model
        if motion_model == MotionModel.CONSTANT_VELOCITY or motion_model == MotionModel.ADAPTIVE:
            self.ndim = 2  # [x, y, vx, vy]
            self.state_dim = 4
        elif motion_model == MotionModel.CONSTANT_ACCELERATION:
            self.ndim = 2  # [x, y, vx, vy, ax, ay]
            self.state_dim = 6
        elif motion_model == MotionModel.CURVILINEAR:
            self.ndim = 2  # [x, y, vx, vy, omega]
            self.state_dim = 5

        # Measurement matrix (we observe position only)
        self._update_mat = np.eye(self.ndim, self.state_dim)

        # Noise parameters (tuned for cattle)
        self._std_weight_position = 1. / 10  # Increased from 1/20 (more uncertainty in position)
        self._std_weight_velocity = 1. / 80  # Increased from 1/160 (more uncertainty in velocity)
        self._std_weight_acceleration = 1. / 160  # Increased (cattle accelerate slowly)
        self._std_weight_angular = 1. / 50  # Increased from 1/100 (more turning uncertainty)

    def initiate(self, measurement):
        """
        Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Position (x, y) in BEV coordinates

        Returns
        -------
        (ndarray, ndarray)
            Mean vector and covariance matrix of the new track
        """
        mean_pos = measurement

        if self.motion_model == MotionModel.CONSTANT_ACCELERATION:
            # [x, y, vx=0, vy=0, ax=0, ay=0]
            mean = np.zeros(6)
            mean[:2] = mean_pos
            std = [
                2 * self._std_weight_position,  # x
                2 * self._std_weight_position,  # y
                10 * self._std_weight_velocity,  # vx
                10 * self._std_weight_velocity,  # vy
                10 * self._std_weight_acceleration,  # ax
                10 * self._std_weight_acceleration  # ay
            ]
        elif self.motion_model == MotionModel.CURVILINEAR:
            # [x, y, vx=0, vy=0, omega=0]
            mean = np.zeros(5)
            mean[:2] = mean_pos
            std = [
                2 * self._std_weight_position,  # x
                2 * self._std_weight_position,  # y
                10 * self._std_weight_velocity,  # vx
                10 * self._std_weight_velocity,  # vy
                10 * self._std_weight_angular  # omega
            ]
        else:  # CONSTANT_VELOCITY or ADAPTIVE
            # [x, y, vx=0, vy=0]
            mean = np.zeros(4)
            mean[:2] = mean_pos
            std = [
                2 * self._std_weight_position,  # x
                2 * self._std_weight_position,  # y
                10 * self._std_weight_velocity,  # vx
                10 * self._std_weight_velocity  # vy
            ]

        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance, adaptive_model=None):
        """
        Run Kalman filter prediction step with non-linear motion.

        Parameters
        ----------
        mean : ndarray
            State mean vector at previous time
        covariance : ndarray
            State covariance matrix at previous time
        adaptive_model : str, optional
            Override motion model for adaptive tracking

        Returns
        -------
        (ndarray, ndarray)
            Predicted mean and covariance
        """
        # Determine which model to use
        active_model = adaptive_model if adaptive_model else self.motion_model

        if active_model == MotionModel.CONSTANT_ACCELERATION or \
                (active_model == MotionModel.ADAPTIVE and len(mean) == 6):
            return self._predict_acceleration(mean, covariance)
        elif active_model == MotionModel.CURVILINEAR:
            return self._predict_curvilinear(mean, covariance)
        else:  # CONSTANT_VELOCITY or ADAPTIVE with 4D state
            return self._predict_constant_velocity(mean, covariance)

    def _predict_constant_velocity(self, mean, covariance):
        """Standard constant velocity prediction"""
        motion_mat = np.eye(4, 4)
        motion_mat[0, 2] = self.dt
        motion_mat[1, 3] = self.dt

        std_pos = [self._std_weight_position, self._std_weight_position]
        std_vel = [self._std_weight_velocity, self._std_weight_velocity]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(motion_mat, mean)
        covariance = np.linalg.multi_dot((
            motion_mat, covariance, motion_mat.T)) + motion_cov

        return mean, covariance

    def _predict_acceleration(self, mean, covariance):
        """
        Constant acceleration model: handles speeding up/slowing down

        x(t+1) = x(t) + vx(t)*dt + 0.5*ax(t)*dt^2
        vx(t+1) = vx(t) + ax(t)*dt
        ax(t+1) = ax(t)
        """
        dt = self.dt

        # Motion matrix for [x, y, vx, vy, ax, ay]
        motion_mat = np.array([
            [1, 0, dt, 0, 0.5 * dt ** 2, 0],
            [0, 1, 0, dt, 0, 0.5 * dt ** 2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])

        # Process noise
        std_pos = [self._std_weight_position] * 2
        std_vel = [self._std_weight_velocity] * 2
        std_acc = [self._std_weight_acceleration] * 2
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel, std_acc]))

        mean = np.dot(motion_mat, mean)
        covariance = np.linalg.multi_dot((
            motion_mat, covariance, motion_mat.T)) + motion_cov

        return mean, covariance

    def _predict_curvilinear(self, mean, covariance):
        """
        Curvilinear motion model: handles turning/circular motion

        For small omega*dt, approximate:
        x(t+1) ≈ x(t) + vx(t)*dt - 0.5*vy(t)*omega*dt^2
        y(t+1) ≈ y(t) + vy(t)*dt + 0.5*vx(t)*omega*dt^2
        vx(t+1) ≈ vx(t) - vy(t)*omega*dt
        vy(t+1) ≈ vy(t) + vx(t)*omega*dt
        omega(t+1) = omega(t)
        """
        dt = self.dt
        x, y, vx, vy, omega = mean

        # Non-linear prediction (using small angle approximation)
        if abs(omega * dt) < 0.1:  # Small turning angle
            # Linear approximation
            new_x = x + vx * dt - 0.5 * vy * omega * dt ** 2
            new_y = y + vy * dt + 0.5 * vx * omega * dt ** 2
            new_vx = vx - vy * omega * dt
            new_vy = vy + vx * omega * dt
            new_omega = omega

            # Jacobian for linearization
            F = np.array([
                [1, 0, dt, -0.5 * omega * dt ** 2, -0.5 * vy * dt ** 2],
                [0, 1, 0.5 * omega * dt ** 2, dt, 0.5 * vx * dt ** 2],
                [0, 0, 1, -omega * dt, -vy * dt],
                [0, 0, omega * dt, 1, vx * dt],
                [0, 0, 0, 0, 1]
            ])
        else:  # Large turning angle - use exact circular motion
            theta = omega * dt
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)

            # Exact circular motion
            new_x = x + (vx * sin_theta + vy * (1 - cos_theta)) / omega
            new_y = y + (vy * sin_theta - vx * (1 - cos_theta)) / omega
            new_vx = vx * cos_theta - vy * sin_theta
            new_vy = vx * sin_theta + vy * cos_theta
            new_omega = omega

            # Jacobian (complex for exact model)
            F = np.array([
                [1, 0, sin_theta / omega, (1 - cos_theta) / omega,
                 (vx * (cos_theta * dt) + vy * sin_theta * dt) / omega -
                 (vx * sin_theta + vy * (1 - cos_theta)) / omega ** 2],
                [0, 1, -(1 - cos_theta) / omega, sin_theta / omega,
                 (vy * cos_theta * dt - vx * sin_theta * dt) / omega -
                 (vy * sin_theta - vx * (1 - cos_theta)) / omega ** 2],
                [0, 0, cos_theta, -sin_theta, -vx * sin_theta * dt - vy * cos_theta * dt],
                [0, 0, sin_theta, cos_theta, vx * cos_theta * dt - vy * sin_theta * dt],
                [0, 0, 0, 0, 1]
            ])

        new_mean = np.array([new_x, new_y, new_vx, new_vy, new_omega])

        # Process noise
        std_pos = [self._std_weight_position] * 2
        std_vel = [self._std_weight_velocity] * 2
        std_ang = [self._std_weight_angular]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel, std_ang]))

        # Extended Kalman Filter covariance update
        covariance = np.linalg.multi_dot((F, covariance, F.T)) + motion_cov

        return new_mean, covariance

    def project(self, mean, covariance):
        """Project state to measurement space (position only)"""
        std = [self._std_weight_position, self._std_weight_position]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))

        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        """Standard Kalman update (works for all models)"""
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T

        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))

        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False, metric='maha'):
        """Compute gating distance (same as original)"""
        mean, covariance = self.project(mean, covariance)

        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean

        if metric == 'gaussian':
            return np.sum(d * d, axis=1)
        elif metric == 'maha':
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(
                cholesky_factor, d.T, lower=True,
                check_finite=False, overwrite_b=True)
            squared_maha = np.sum(z * z, axis=0)
            return squared_maha
        else:
            raise ValueError('invalid distance metric')

    def multi_predict(self, mean, covariance):
        """Vectorized prediction for multiple tracks"""
        n_tracks = len(mean)
        predicted_mean = np.zeros_like(mean)
        predicted_cov = np.zeros_like(covariance)

        for i in range(n_tracks):
            predicted_mean[i], predicted_cov[i] = self.predict(
                mean[i], covariance[i])

        return predicted_mean, predicted_cov