"""
Extended Kalman Filter for drone state estimation.

Fuses IMU predictions with vision-based gate pose measurements
to maintain accurate state estimates at high frequency (500 Hz)
despite low-frequency vision updates (24 Hz).
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
from enum import Enum


class MeasurementType(Enum):
    """Types of measurements the EKF can process."""
    GATE_POSE = "gate_pose"
    IMU = "imu"
    OPTICAL_FLOW = "optical_flow"


@dataclass
class EKFState:
    """Full EKF state representation."""
    # Position in world frame [m]
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # Velocity in world frame [m/s]
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # Orientation as quaternion (w, x, y, z)
    orientation: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0.]))
    # Angular velocity in body frame [rad/s]
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # Accelerometer bias [m/s^2]
    accel_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # Gyroscope bias [rad/s]
    gyro_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def as_vector(self) -> np.ndarray:
        """Convert state to flat vector (19 elements)."""
        return np.concatenate([
            self.position,       # 3
            self.velocity,       # 3
            self.orientation,    # 4
            self.angular_velocity,  # 3
            self.accel_bias,     # 3
            self.gyro_bias,      # 3
        ])

    @classmethod
    def from_vector(cls, vec: np.ndarray) -> "EKFState":
        """Create state from flat vector."""
        return cls(
            position=vec[0:3].copy(),
            velocity=vec[3:6].copy(),
            orientation=vec[6:10].copy(),
            angular_velocity=vec[10:13].copy(),
            accel_bias=vec[13:16].copy(),
            gyro_bias=vec[16:19].copy(),
        )

    @property
    def rotation_matrix(self) -> np.ndarray:
        """Get rotation matrix from orientation quaternion."""
        w, x, y, z = self.orientation
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
        ])

    @property
    def euler_angles(self) -> np.ndarray:
        """Get Euler angles (roll, pitch, yaw) from quaternion."""
        w, x, y, z = self.orientation

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.array([roll, pitch, yaw])


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for drone state estimation.

    State vector (19 elements):
    - Position (3): x, y, z in world frame
    - Velocity (3): vx, vy, vz in world frame
    - Orientation (4): quaternion (w, x, y, z)
    - Angular velocity (3): wx, wy, wz in body frame
    - Accelerometer bias (3): bax, bay, baz
    - Gyroscope bias (3): bgx, bgy, bgz

    The filter runs prediction at high frequency (500 Hz) using IMU
    data and updates at low frequency (24 Hz) when vision measurements
    are available.
    """

    # State dimensions
    STATE_DIM = 19
    POS_IDX = slice(0, 3)
    VEL_IDX = slice(3, 6)
    ORI_IDX = slice(6, 10)
    ANG_VEL_IDX = slice(10, 13)
    ACCEL_BIAS_IDX = slice(13, 16)
    GYRO_BIAS_IDX = slice(16, 19)

    def __init__(
        self,
        extrinsic_matrix: Optional[np.ndarray] = None,
        # Process noise parameters
        process_noise_pos: float = 0.01,
        process_noise_vel: float = 0.1,
        process_noise_ori: float = 0.01,
        process_noise_ang_vel: float = 0.1,
        process_noise_accel_bias: float = 0.001,
        process_noise_gyro_bias: float = 0.001,
        # Measurement noise parameters
        measurement_noise_gate_pos: float = 0.05,
        measurement_noise_gate_ori: float = 0.1,
        # Initial covariance
        initial_cov_pos: float = 0.1,
        initial_cov_vel: float = 0.5,
        initial_cov_ori: float = 0.1,
        initial_cov_ang_vel: float = 0.5,
        initial_cov_bias: float = 0.01,
        # Physical constants
        gravity: float = 9.81,
    ):
        """
        Initialize EKF.

        Args:
            process_noise_*: Process noise standard deviations
            measurement_noise_*: Measurement noise standard deviations
            initial_cov_*: Initial covariance diagonal elements
            gravity: Gravitational acceleration [m/s^2]
        """
        self.gravity = np.array([0, 0, -gravity])
        self.R_cam_to_body = np.array([
            [0, 0, 1],
            [-1, 0, 0],
            [0, -1, 0],
        ]) if extrinsic_matrix is None else np.linalg.inv(extrinsic_matrix)

        # Store noise parameters
        self.process_noise = {
            "pos": process_noise_pos,
            "vel": process_noise_vel,
            "ori": process_noise_ori,
            "ang_vel": process_noise_ang_vel,
            "accel_bias": process_noise_accel_bias,
            "gyro_bias": process_noise_gyro_bias,
        }

        self.measurement_noise = {
            "gate_pos": measurement_noise_gate_pos,
            "gate_ori": measurement_noise_gate_ori,
        }

        # Initialize state
        self.state = EKFState()

        # Initialize covariance matrix
        self.P = np.diag([
            initial_cov_pos, initial_cov_pos, initial_cov_pos,  # position
            initial_cov_vel, initial_cov_vel, initial_cov_vel,  # velocity
            initial_cov_ori, initial_cov_ori, initial_cov_ori, initial_cov_ori,  # orientation
            initial_cov_ang_vel, initial_cov_ang_vel, initial_cov_ang_vel,  # angular velocity
            initial_cov_bias, initial_cov_bias, initial_cov_bias,  # accel bias
            initial_cov_bias, initial_cov_bias, initial_cov_bias,  # gyro bias
        ])

        # Process noise matrix (computed in predict)
        self.Q = None

        # Measurement noise matrices
        self.R_gate = np.diag([
            measurement_noise_gate_pos**2,
            measurement_noise_gate_pos**2,
            measurement_noise_gate_pos**2,
            measurement_noise_gate_ori**2,
            measurement_noise_gate_ori**2,
            measurement_noise_gate_ori**2,
        ])

        # Known gate positions for measurement updates
        self.known_gates: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

        # Tracking
        self.current_gate_idx = 0
        self.last_update_time = 0.0
        self.prediction_count = 0
        self.update_count = 0

    def set_known_gates(self, gates: Dict[int, Tuple[np.ndarray, np.ndarray]]):
        """
        Set known gate positions and orientations.

        Args:
            gates: Dict mapping gate_id to (position, orientation) tuples
                   where orientation is quaternion (w, x, y, z)
        """
        self.known_gates = gates

    def reset(
        self,
        position: Optional[np.ndarray] = None,
        velocity: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ):
        """
        Reset filter state.

        Args:
            position: Initial position [x, y, z]
            velocity: Initial velocity [vx, vy, vz]
            orientation: Initial orientation quaternion (w, x, y, z)
        """
        self.state = EKFState()

        if position is not None:
            self.state.position = position.copy()
        if velocity is not None:
            self.state.velocity = velocity.copy()
        if orientation is not None:
            self.state.orientation = orientation.copy()

        # Reset covariance to initial values
        self.P = np.eye(self.STATE_DIM) * 0.1

        self.prediction_count = 0
        self.update_count = 0

    def predict(
        self,
        dt: float,
        accel: Optional[np.ndarray] = None,
        gyro: Optional[np.ndarray] = None,
    ) -> EKFState:
        """
        Prediction step using IMU data or constant velocity model.

        Args:
            dt: Time step [s]
            accel: Accelerometer reading in body frame [m/s^2]
            gyro: Gyroscope reading in body frame [rad/s]

        Returns:
            Predicted state
        """
        if dt <= 0:
            return self.state

        # Get rotation matrix
        R = self.state.rotation_matrix

        if accel is not None and gyro is not None:
            # Remove bias from measurements
            accel_corrected = accel - self.state.accel_bias
            gyro_corrected = gyro - self.state.gyro_bias

            # Transform acceleration to world frame and add gravity
            accel_world = R @ accel_corrected + self.gravity

            # Update velocity and position
            self.state.velocity = self.state.velocity + accel_world * dt
            self.state.position = self.state.position + self.state.velocity * dt + 0.5 * accel_world * dt**2

            # Update orientation using gyro
            self.state.angular_velocity = gyro_corrected
            self.state.orientation = self._integrate_quaternion(
                self.state.orientation,
                gyro_corrected,
                dt
            )
        else:
            # Constant velocity model
            self.state.position = self.state.position + self.state.velocity * dt

        # Normalize quaternion
        self.state.orientation = self.state.orientation / np.linalg.norm(self.state.orientation)

        # Compute process noise matrix Q
        Q = self._compute_process_noise(dt)

        # Compute state transition Jacobian F
        F = self._compute_state_jacobian(dt, accel, gyro)

        # Propagate covariance: P = F @ P @ F.T + Q
        self.P = F @ self.P @ F.T + Q

        self.prediction_count += 1
        return self.state

    def update_gate_pose(
        self,
        gate_position_camera: np.ndarray,
        gate_orientation_camera: np.ndarray,
        gate_idx: int,
    ) -> EKFState:
        """
        Update state using gate pose measurement in camera frame.

        Args:
            gate_position_camera: Gate position in camera frame [x, y, z]
            gate_orientation_camera: Gate orientation in camera frame (quaternion)
            gate_idx: Index of the detected gate

        Returns:
            Updated state
        """
        if gate_idx not in self.known_gates:
            return self.state

        known_pos, known_ori = self.known_gates[gate_idx]

        # Transform gate observation to world frame estimate of drone position
        # If we see gate at position p_c in camera frame,
        # then drone position = gate_world_pos - R_world_cam @ p_c

        R_cam_to_world = self.state.rotation_matrix @ self.R_cam_to_body

        # Camera frame to world frame (camera is forward-looking)
        # Camera: z forward, x right, y down
        # World: x forward, y left, z up

        # Gate position in world frame (from observation)
        # gate_pos_body = self.R_cam_to_body @ gate_position_camera
        gate_pos_world = R_cam_to_world @ gate_position_camera + self.state.position

        # Measurement: expected vs observed gate position
        z = gate_pos_world  # Observed gate position in world
        h = known_pos  # Expected gate position (from map)

        # Innovation
        y = h - z

        # Measurement Jacobian H (partial derivatives of measurement wrt state)
        # Simplified: measurement is gate position which depends on drone position
        H = np.zeros((3, self.STATE_DIM))
        H[0:3, self.POS_IDX] = np.eye(3)

        # Measurement noise
        R_noise = np.eye(3) * self.measurement_noise["gate_pos"]**2

        # Kalman gain
        S = H @ self.P @ H.T + R_noise
        K = self.P @ H.T @ np.linalg.inv(S)

        # State update
        state_vec = self.state.as_vector()
        state_vec = state_vec + K @ y
        self.state = EKFState.from_vector(state_vec)

        # Normalize quaternion
        self.state.orientation = self.state.orientation / np.linalg.norm(self.state.orientation)

        # Covariance update
        I = np.eye(self.STATE_DIM)
        self.P = (I - K @ H) @ self.P

        self.update_count += 1
        return self.state

    def update_vision(
        self,
        drone_position_estimate: np.ndarray,
        confidence: float = 1.0,
    ) -> EKFState:
        """
        Direct update using vision-based position estimate.

        Args:
            drone_position_estimate: Estimated drone position in world frame
            confidence: Measurement confidence [0, 1]

        Returns:
            Updated state
        """
        # Measurement
        z = drone_position_estimate

        # Expected measurement (current state)
        h = self.state.position

        # Innovation
        y = z - h

        # Measurement Jacobian
        H = np.zeros((3, self.STATE_DIM))
        H[0:3, self.POS_IDX] = np.eye(3)

        # Adjust measurement noise based on confidence
        base_noise = self.measurement_noise["gate_pos"]
        adjusted_noise = base_noise / (confidence + 0.1)
        R_noise = np.eye(3) * adjusted_noise**2

        # Kalman gain
        S = H @ self.P @ H.T + R_noise
        K = self.P @ H.T @ np.linalg.inv(S)

        # State update
        state_vec = self.state.as_vector()
        state_vec = state_vec + K @ y
        self.state = EKFState.from_vector(state_vec)

        # Normalize quaternion
        self.state.orientation = self.state.orientation / np.linalg.norm(self.state.orientation)

        # Covariance update (Joseph form for numerical stability)
        I = np.eye(self.STATE_DIM)
        IKH = I - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R_noise @ K.T

        self.update_count += 1
        return self.state

    def _integrate_quaternion(
        self,
        q: np.ndarray,
        omega: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """
        Integrate quaternion using angular velocity.

        Args:
            q: Current quaternion (w, x, y, z)
            omega: Angular velocity in body frame [rad/s]
            dt: Time step [s]

        Returns:
            Updated quaternion
        """
        # Quaternion derivative: q_dot = 0.5 * q * omega_quat
        omega_quat = np.array([0, omega[0], omega[1], omega[2]])

        # Quaternion multiplication
        q_dot = 0.5 * self._quaternion_multiply(q, omega_quat)

        # First-order integration
        q_new = q + q_dot * dt

        # Normalize
        return q_new / np.linalg.norm(q_new)

    def _quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions (w, x, y, z)."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])

    def _compute_process_noise(self, dt: float) -> np.ndarray:
        """Compute process noise covariance matrix Q."""
        Q = np.zeros((self.STATE_DIM, self.STATE_DIM))

        # Position noise (integrated velocity noise)
        Q[self.POS_IDX, self.POS_IDX] = np.eye(3) * (self.process_noise["pos"] * dt)**2

        # Velocity noise
        Q[self.VEL_IDX, self.VEL_IDX] = np.eye(3) * (self.process_noise["vel"] * dt)**2

        # Orientation noise
        Q[self.ORI_IDX, self.ORI_IDX] = np.eye(4) * (self.process_noise["ori"] * dt)**2

        # Angular velocity noise
        Q[self.ANG_VEL_IDX, self.ANG_VEL_IDX] = np.eye(3) * (self.process_noise["ang_vel"] * dt)**2

        # Bias noise (random walk)
        Q[self.ACCEL_BIAS_IDX, self.ACCEL_BIAS_IDX] = np.eye(3) * (self.process_noise["accel_bias"] * dt)**2
        Q[self.GYRO_BIAS_IDX, self.GYRO_BIAS_IDX] = np.eye(3) * (self.process_noise["gyro_bias"] * dt)**2

        return Q

    def _compute_state_jacobian(
        self,
        dt: float,
        accel: Optional[np.ndarray] = None,
        gyro: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute state transition Jacobian F."""
        F = np.eye(self.STATE_DIM)

        # Position depends on velocity
        F[self.POS_IDX, self.VEL_IDX] = np.eye(3) * dt

        # Velocity depends on orientation (through rotation of acceleration)
        # Simplified: small angle approximation
        F[self.VEL_IDX, self.VEL_IDX] = np.eye(3)

        # Orientation evolution (linearized quaternion dynamics)
        # F_q approximately identity for small dt
        F[self.ORI_IDX, self.ORI_IDX] = np.eye(4)

        return F

    def get_state_covariance(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get position, velocity, and orientation covariance.

        Returns:
            Tuple of (position_cov, velocity_cov, orientation_cov)
        """
        pos_cov = self.P[self.POS_IDX, self.POS_IDX]
        vel_cov = self.P[self.VEL_IDX, self.VEL_IDX]
        ori_cov = self.P[self.ORI_IDX, self.ORI_IDX]
        return pos_cov, vel_cov, ori_cov

    def get_state_uncertainty(self) -> Dict[str, float]:
        """
        Get state uncertainty (standard deviation).

        Returns:
            Dict with position, velocity, orientation uncertainties
        """
        pos_cov, vel_cov, ori_cov = self.get_state_covariance()
        return {
            "position": np.sqrt(np.trace(pos_cov) / 3),
            "velocity": np.sqrt(np.trace(vel_cov) / 3),
            "orientation": np.sqrt(np.trace(ori_cov) / 4),
        }


if __name__ == "__main__":
    # Test EKF
    print("Testing Extended Kalman Filter...")

    ekf = ExtendedKalmanFilter()

    # Set initial state
    ekf.reset(
        position=np.array([0.0, 0.0, 1.0]),
        velocity=np.array([1.0, 0.0, 0.0]),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
    )

    # Register a known gate
    ekf.set_known_gates({
        0: (np.array([5.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0, 0.0]))
    })

    print(f"Initial state: pos={ekf.state.position}, vel={ekf.state.velocity}")

    # Simulate motion
    dt = 0.002  # 500 Hz
    for i in range(500):  # 1 second
        # Predict with simulated IMU
        accel = np.array([0.0, 0.0, 9.81])  # Hover
        gyro = np.array([0.0, 0.0, 0.0])
        ekf.predict(dt, accel, gyro)

        # Vision update at 24 Hz (every ~21 steps)
        if i % 21 == 0:
            # Simulate vision measurement with some noise
            true_pos = np.array([i * dt * 1.0, 0.0, 1.0])
            noisy_pos = true_pos + np.random.normal(0, 0.02, 3)
            ekf.update_vision(noisy_pos, confidence=0.8)

    print(f"\nAfter 1 second:")
    print(f"Position: {ekf.state.position}")
    print(f"Velocity: {ekf.state.velocity}")
    print(f"Predictions: {ekf.prediction_count}")
    print(f"Updates: {ekf.update_count}")
    print(f"Uncertainty: {ekf.get_state_uncertainty()}")

    print("\nTest complete!")
