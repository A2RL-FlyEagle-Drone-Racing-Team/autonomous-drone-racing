"""
Vision Racing Pipeline - End-to-End Integration.

Coordinates all components for vision-based drone racing:
Camera (24Hz) -> GateNet -> QuAdGate -> EKF (500Hz) -> G&CNet -> Motors

This pipeline handles the asynchronous nature of vision updates (24 Hz)
with high-frequency control (500 Hz) using the Extended Kalman Filter
for state estimation.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import time

import torch
import torch.backends.mps

# Import our modules
from ..vision.gate_net import GateNet, create_gatenet
from ..vision.quad_gate import QuAdGate, GateTracker, GateDetection
from ..vision.pose_estimator import PoseEstimator, GatePose
from ..state.ekf import ExtendedKalmanFilter, EKFState
from ..control.gcnet import GCNet, create_gcnet
from ..control.motor_mixer import MotorMixer, DroneParams


@dataclass
class PipelineConfig:
    """Configuration for the vision racing pipeline."""
    # Frequencies
    control_freq: int = 500  # Hz
    vision_freq: int = 24  # Hz

    # Gate parameters
    gate_width: float = 1.0
    gate_height: float = 1.0

    # Camera parameters
    image_width: int = 64
    image_height: int = 48
    camera_fov: float = 60.0

    # Control parameters
    target_velocity: float = 5.0  # m/s
    # MAX_RPM from gym-pybullet-drones Crazyflie 2.x model (~21702)
    # Based on: sqrt((thrust_to_weight * mass * g) / (4 * kf))
    max_rpm: float = 21702.64

    # Model paths
    gate_net_path: Optional[str] = None
    gcnet_path: Optional[str] = None

    # Device
    device: str = "auto"


class VisionRacingPipeline:
    """
    End-to-end vision-based racing pipeline.

    Workflow:
    1. Receive camera image at 24 Hz
    2. Process through GateNet to get segmentation mask
    3. Extract corners with QuAdGate
    4. Estimate gate pose with PnP
    5. Update EKF with gate observation
    6. Run EKF prediction at 500 Hz
    7. Generate motor commands with G&CNet

    The pipeline maintains state across frames and handles
    the mismatch between vision update rate and control rate.
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        known_gates: Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]] = None,
    ):
        """
        Initialize pipeline.

        Args:
            config: Pipeline configuration
            known_gates: Dict of gate_id -> (position, orientation) for EKF updates
        """
        self.config = config or PipelineConfig()

        # Initialize device
        if self.config.device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(self.config.device)

        # Initialize components
        self._init_vision()
        self._init_state_estimation(known_gates)
        self._init_control()

        # Timing
        self.control_dt = 1.0 / self.config.control_freq
        self.vision_dt = 1.0 / self.config.vision_freq
        self.last_vision_time = 0.0
        self.last_control_time = 0.0

        # State tracking
        self.current_gate_idx = 0
        self.gates_passed = 0
        self.total_steps = 0

        # Latest observations
        self.latest_detection: Optional[GateDetection] = None
        self.latest_pose: Optional[GatePose] = None
        self.latest_image: Optional[np.ndarray] = None

    def _init_vision(self):
        """Initialize vision components."""
        # GateNet for segmentation
        if self.config.gate_net_path:
            self.gate_net = create_gatenet(self.config.gate_net_path, device=str(self.device))
        else:
            self.gate_net = GateNet().to(self.device)
            self.gate_net.eval()

        # QuAdGate for corner detection
        self.quad_gate = QuAdGate()
        self.gate_tracker = GateTracker(self.quad_gate)

        # Pose estimator
        self.pose_estimator = PoseEstimator(
            gate_width=self.config.gate_width,
            gate_height=self.config.gate_height,
            image_size=(self.config.image_width, self.config.image_height),
            camera_fov=self.config.camera_fov,
        )

    def _init_state_estimation(
        self,
        known_gates: Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]] = None,
    ):
        """Initialize state estimation (EKF)."""
        self.ekf = ExtendedKalmanFilter()

        if known_gates:
            self.ekf.set_known_gates(known_gates)

        self.known_gates = known_gates or {}

    def _init_control(self):
        """Initialize control components."""
        # G&CNet controller
        if self.config.gcnet_path:
            self.gcnet = create_gcnet(self.config.gcnet_path, device=str(self.device))
        else:
            self.gcnet = GCNet().to(self.device)
            self.gcnet.eval()

        # Motor mixer (for fallback PID control)
        self.motor_mixer = MotorMixer()

    def reset(
        self,
        initial_position: Optional[np.ndarray] = None,
        initial_velocity: Optional[np.ndarray] = None,
        initial_orientation: Optional[np.ndarray] = None,
    ):
        """
        Reset pipeline state.

        Args:
            initial_position: Starting position [x, y, z]
            initial_velocity: Starting velocity [vx, vy, vz]
            initial_orientation: Starting orientation (w, x, y, z)
        """
        # Reset EKF
        self.ekf.reset(
            position=initial_position,
            velocity=initial_velocity,
            orientation=initial_orientation,
        )

        # Reset tracking
        self.gate_tracker.reset()
        self.current_gate_idx = 0
        self.gates_passed = 0
        self.total_steps = 0

        # Reset timing
        self.last_vision_time = 0.0
        self.last_control_time = 0.0

        # Clear latest observations
        self.latest_detection = None
        self.latest_pose = None
        self.latest_image = None

    def process_image(
        self,
        rgb_image: np.ndarray,
        current_time: float,
    ) -> Optional[GatePose]:
        """
        Process camera image through vision pipeline.

        Args:
            rgb_image: RGB image (H, W, 3) or (H, W, 4)
            current_time: Current simulation time

        Returns:
            Gate pose if detected, None otherwise
        """
        self.latest_image = rgb_image
        self.last_vision_time = current_time

        # Preprocess image
        if rgb_image.shape[-1] == 4:
            rgb_image = rgb_image[..., :3]

        # Convert to tensor (B, C, H, W)
        img_tensor = torch.from_numpy(rgb_image).float().permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor / 255.0
        img_tensor = img_tensor.to(self.device)

        # Get segmentation mask
        with torch.no_grad():
            mask = self.gate_net(img_tensor)

        mask_np = mask.squeeze().cpu().numpy()

        # Detect corners
        detection = self.gate_tracker.update(mask_np)
        self.latest_detection = detection

        if detection is None or detection.confidence < 0.3:
            return None

        # Estimate pose
        pose = self.pose_estimator.estimate_pose(detection)
        self.latest_pose = pose

        return pose

    def update_state_with_vision(
        self,
        gate_pose: GatePose,
        gate_idx: int,
    ):
        """
        Update EKF state with vision measurement.

        Args:
            gate_pose: Detected gate pose in camera frame
            gate_idx: Index of detected gate
        """
        if gate_pose is None:
            return

        # Update EKF with gate pose measurement
        self.ekf.update_gate_pose(
            gate_position_camera=gate_pose.position,
            gate_orientation_camera=gate_pose.orientation,
            gate_idx=gate_idx,
        )

    def predict_state(
        self,
        dt: float,
        accel: Optional[np.ndarray] = None,
        gyro: Optional[np.ndarray] = None,
    ) -> EKFState:
        """
        Run EKF prediction step.

        Args:
            dt: Time step
            accel: Accelerometer reading (optional)
            gyro: Gyroscope reading (optional)

        Returns:
            Predicted state
        """
        return self.ekf.predict(dt, accel, gyro)

    def compute_motor_commands(
        self,
        state: Optional[EKFState] = None,
    ) -> np.ndarray:
        """
        Compute motor RPM commands using G&CNet.

        Args:
            state: Current state estimate (uses EKF state if None)

        Returns:
            4-element array of motor RPMs
        """
        if state is None:
            state = self.ekf.state

        # Get current gate target
        if self.current_gate_idx in self.known_gates:
            gate_pos, gate_ori = self.known_gates[self.current_gate_idx]
        else:
            # Fallback: use placeholder
            gate_pos = np.array([5, 0, 1])
            gate_ori = np.array([1, 0, 0, 0])

        # Compute gate direction
        R_gate = self._quat_to_rotation_matrix(gate_ori)
        gate_dir = R_gate[:, 0]  # Gate normal

        # Compute velocity target
        to_gate = gate_pos - state.position
        to_gate_norm = to_gate / (np.linalg.norm(to_gate) + 1e-6)
        vel_target = to_gate_norm * self.config.target_velocity

        # Get action from G&CNet
        rpms = self.gcnet.get_action(
            position=state.position,
            velocity=state.velocity,
            orientation=state.orientation,  # Already (w, x, y, z)
            angular_velocity=state.angular_velocity,
            gate_position=gate_pos,
            gate_direction=gate_dir,
            velocity_target=vel_target,
        )

        return rpms

    def compute_from_ground_truth(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        orientation: np.ndarray,
        angular_velocity: np.ndarray,
    ) -> np.ndarray:
        """
        Compute motor commands from ground truth state (bypasses EKF).

        Args:
            position: World position [x, y, z]
            velocity: World velocity [vx, vy, vz]
            orientation: Quaternion (w, x, y, z)
            angular_velocity: Body angular rates [wx, wy, wz]

        Returns:
            Motor RPM commands
        """
        # Get current gate target
        if self.current_gate_idx in self.known_gates:
            gate_pos, gate_ori = self.known_gates[self.current_gate_idx]
        else:
            gate_pos = np.array([5, 0, 1])
            gate_ori = np.array([1, 0, 0, 0])

        # Compute gate direction
        R_gate = self._quat_to_rotation_matrix(gate_ori)
        gate_dir = R_gate[:, 0]

        # Compute velocity target
        to_gate = gate_pos - position
        to_gate_norm = to_gate / (np.linalg.norm(to_gate) + 1e-6)
        vel_target = to_gate_norm * self.config.target_velocity

        # Check gate progress
        dist_to_gate = np.linalg.norm(position - gate_pos)
        if dist_to_gate < 0.5:
            self._advance_gate()

        # Get action from G&CNet
        rpms = self.gcnet.get_action(
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
            gate_position=gate_pos,
            gate_direction=gate_dir,
            velocity_target=vel_target,
        )

        return rpms

    def step(
        self,
        current_time: float,
        rgb_image: Optional[np.ndarray] = None,
        accel: Optional[np.ndarray] = None,
        gyro: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Run one control step of the pipeline.

        Args:
            current_time: Current simulation time
            rgb_image: Camera image (None if no new image available)
            accel: Accelerometer reading
            gyro: Gyroscope reading

        Returns:
            Tuple of (motor_rpms, info_dict)
        """
        self.total_steps += 1

        # Compute dt since last control
        dt = current_time - self.last_control_time
        if dt <= 0:
            dt = self.control_dt
        self.last_control_time = current_time

        # Process vision if new image available
        gate_pose = None
        if rgb_image is not None:
            gate_pose = self.process_image(rgb_image, current_time)
            if gate_pose is not None:
                self.update_state_with_vision(gate_pose, self.current_gate_idx)

        # EKF prediction
        state = self.predict_state(dt, accel, gyro)

        # Check gate progress
        if self.current_gate_idx in self.known_gates:
            gate_pos, _ = self.known_gates[self.current_gate_idx]
            dist_to_gate = np.linalg.norm(state.position - gate_pos)
            if dist_to_gate < 0.5:  # Gate tolerance
                self._advance_gate()

        # Compute motor commands
        rpms = self.compute_motor_commands(state)

        # Build info dict
        info = {
            "position": state.position.copy(),
            "velocity": state.velocity.copy(),
            "orientation": state.orientation.copy(),
            "gates_passed": self.gates_passed,
            "current_gate": self.current_gate_idx,
            "state_uncertainty": self.ekf.get_state_uncertainty(),
            "vision_detected": gate_pose is not None,
        }

        return rpms, info

    def _advance_gate(self):
        """Advance to next gate."""
        self.gates_passed += 1
        self.current_gate_idx = (self.current_gate_idx + 1) % len(self.known_gates)

    def _quat_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion (w, x, y, z) to rotation matrix."""
        w, x, y, z = q / np.linalg.norm(q)
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
        ])

    def get_state(self) -> EKFState:
        """Get current state estimate."""
        return self.ekf.state

    def set_gate_index(self, idx: int):
        """Manually set current gate index."""
        self.current_gate_idx = idx

    def load_models(
        self,
        gate_net_path: Optional[str] = None,
        gcnet_path: Optional[str] = None,
    ):
        """
        Load pretrained models.

        Args:
            gate_net_path: Path to GateNet checkpoint
            gcnet_path: Path to G&CNet checkpoint
        """
        if gate_net_path:
            self.gate_net = create_gatenet(gate_net_path, device=str(self.device))

        if gcnet_path:
            self.gcnet = create_gcnet(gcnet_path, device=str(self.device))


class PipelineRunner:
    """
    Helper class to run the pipeline with a gym environment.
    """

    def __init__(
        self,
        pipeline: VisionRacingPipeline,
        env: Any,  # HighFreqRacingAviary
    ):
        """
        Initialize runner.

        Args:
            pipeline: VisionRacingPipeline instance
            env: Environment instance
        """
        self.pipeline = pipeline
        self.env = env

    def run_episode(
        self,
        max_steps: int = 1000,
        render: bool = False,
    ) -> Dict[str, Any]:
        """
        Run one complete episode.

        Args:
            max_steps: Maximum steps per episode
            render: Whether to render (if GUI available)

        Returns:
            Episode statistics
        """
        obs, info = self.env.reset()

        # Get initial state from environment
        state_dict = self.env.get_state_for_control()
        self.pipeline.reset(
            initial_position=state_dict["position"],
            initial_velocity=state_dict["velocity"],
            initial_orientation=state_dict["orientation"],
        )

        total_reward = 0
        step = 0
        vision_frame_counter = 0
        vision_interval = self.pipeline.config.control_freq // self.pipeline.config.vision_freq

        episode_data = {
            "positions": [],
            "velocities": [],
            "rpms": [],
            "rewards": [],
        }

        while step < max_steps:
            current_time = step * self.pipeline.control_dt

            # Get ground truth state from environment
            state_dict = self.env.get_state_for_control()

            # Use ground truth state directly (bypass EKF for testing)
            rpms = self.pipeline.compute_from_ground_truth(
                position=state_dict["position"],
                velocity=state_dict["velocity"],
                orientation=state_dict["orientation"],
                angular_velocity=state_dict["angular_velocity"],
            )

            # Normalize RPMs to [0, 1] for environment
            action = rpms / self.pipeline.config.max_rpm
            action = np.clip(action, 0, 1)

            # Step environment
            obs, reward, terminated, truncated, env_info = self.env.step(action)
            total_reward += reward

            # Record data
            episode_data["positions"].append(state_dict["position"].copy())
            episode_data["velocities"].append(state_dict["velocity"].copy())
            episode_data["rpms"].append(rpms)
            episode_data["rewards"].append(reward)

            step += 1

            if terminated or truncated:
                break

        # Compute statistics
        positions = np.array(episode_data["positions"])
        velocities = np.array(episode_data["velocities"])

        return {
            "total_reward": total_reward,
            "steps": step,
            "gates_passed": self.pipeline.gates_passed,
            "avg_speed": np.mean(np.linalg.norm(velocities, axis=1)) if len(velocities) > 0 else 0.0,
            "max_speed": np.max(np.linalg.norm(velocities, axis=1)) if len(velocities) > 0 else 0.0,
            "final_position": positions[-1] if len(positions) > 0 else None,
            "episode_data": episode_data,
        }


if __name__ == "__main__":
    # Test pipeline initialization
    print("Testing VisionRacingPipeline...")

    # Create known gates
    known_gates = {
        0: (np.array([2, 0, 1]), np.array([1, 0, 0, 0])),
        1: (np.array([4, 2, 1.2]), np.array([0.92, 0, 0, 0.38])),
        2: (np.array([4, 4, 1]), np.array([0.71, 0, 0, 0.71])),
    }

    config = PipelineConfig(
        control_freq=500,
        vision_freq=24,
        device="cpu",
    )

    pipeline = VisionRacingPipeline(config=config, known_gates=known_gates)

    print(f"Pipeline initialized:")
    print(f"  Control freq: {config.control_freq} Hz")
    print(f"  Vision freq: {config.vision_freq} Hz")
    print(f"  Device: {pipeline.device}")
    print(f"  Known gates: {len(known_gates)}")

    # Reset with initial state
    pipeline.reset(
        initial_position=np.array([0, 0, 1]),
        initial_velocity=np.array([1, 0, 0]),
        initial_orientation=np.array([1, 0, 0, 0]),
    )

    # Test step without vision
    print("\nTesting control steps...")
    for i in range(10):
        rpms, info = pipeline.step(
            current_time=i * 0.002,
            rgb_image=None,
        )
        if i % 5 == 0:
            print(f"  Step {i}: RPMs={rpms.astype(int)}, pos={info['position']}")

    # Test step with mock vision
    print("\nTesting with mock vision...")
    fake_image = np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)
    rpms, info = pipeline.step(
        current_time=0.1,
        rgb_image=fake_image,
    )
    print(f"  Vision detected: {info['vision_detected']}")

    print(f"\nFinal state:")
    print(f"  Position: {pipeline.get_state().position}")
    print(f"  Velocity: {pipeline.get_state().velocity}")
    print(f"  Gates passed: {pipeline.gates_passed}")

    print("\nTest complete!")
