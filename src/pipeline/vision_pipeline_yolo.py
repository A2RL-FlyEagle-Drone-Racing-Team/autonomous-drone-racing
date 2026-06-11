"""
Vision Racing Pipeline with YOLO - End-to-End Integration.

Coordinates vision components for vision-based drone racing using YOLO:
Camera (24Hz) -> YOLO Segmentation -> QuAdGate -> EKF (500Hz)

This pipeline handles the asynchronous nature of vision updates (24 Hz)
with high-frequency state estimation (500 Hz) using the Extended Kalman Filter.
"""

import os
import cv2
import numpy as np
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass

import torch
import torch.backends.mps

# Import YOLO from ultralytics
from ultralytics import YOLO

# Import our modules
from ..vision.quad_gate import QuAdGate, GateTracker, GateDetection
from ..vision.pose_estimator import PoseEstimator, GatePose
from ..state.ekf import ExtendedKalmanFilter, EKFState


@dataclass
class YOLOPipelineConfig:
    """Configuration for the vision racing pipeline."""

    # Frequencies
    vision_freq: int = 60  # Hz

    # Gate parameters
    gate_width: float = 2.7
    gate_height: float = 2.7

    # Camera parameters
    image_width: int = 640  # YOLO default input size
    image_height: int = 480
    camera_fov: Optional[float] = 60.0
    intrinsic_matrix: Optional[np.ndarray] = None
    extrinsic_matrix: Optional[np.ndarray] = None
    dist_coeffs: Optional[np.ndarray] = None

    # Model filename (will be searched in models/ directory)
    yolo_model_name: Optional[str] = "best-seg.engine"
    convert_to_rgb: bool = False

    # Device
    device: str = "auto"

    @classmethod
    def from_kalibr(cls, kalibr_config: Dict[str, Any]) -> "YOLOPipelineConfig":
        """Create config from Kalibr config.
        
        Kalibr stores intrinsics as [fx, fy, cx, cy], this converts to 3x3 camera matrix.
        """
        config = cls()
        config.camera_fov = None
        config.image_width, config.image_height = kalibr_config["cam0"]["resolution"]
        
        # Convert Kalibr intrinsics [fx, fy, cx, cy] to 3x3 camera matrix
        intrinsics = kalibr_config["cam0"]["intrinsics"]
        fx, fy, cx, cy = intrinsics
        config.intrinsic_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ], dtype=np.float64)
        config.extrinsic_matrix = np.array(kalibr_config["cam0"]["T_cam_imu"], dtype=np.float64)[:3, :3]
        
        config.dist_coeffs = np.array(kalibr_config["cam0"]["distortion_coeffs"])
        return config


class VisionRacingYOLOPipeline:
    """
    End-to-end vision-based racing pipeline using YOLO for segmentation.

    Workflow:
    1. Receive camera image at 24 Hz
    2. Process through YOLO to get segmentation mask
    3. Extract corners with QuAdGate
    4. Estimate gate pose with PnP
    5. Update EKF with gate observation
    6. Run EKF prediction at 500 Hz

    The pipeline maintains state across frames and handles
    the mismatch between vision update rate and state estimation rate.
    """

    def __init__(
        self,
        config: Optional[YOLOPipelineConfig] = None,
        known_gates: Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]] = None,
    ):
        """
        Initialize pipeline.

        Args:
            config: Pipeline configuration
            known_gates: Dict of gate_id -> (position, orientation) for EKF updates
        """
        self.config = config or YOLOPipelineConfig()

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

        # Timing
        self.vision_dt = 1.0 / self.config.vision_freq
        self.last_vision_time = 0.0

        # State tracking
        self.current_gate_idx = 0
        self.gates_passed = 0
        self.total_steps = 0

        # Latest observations
        self.latest_detection: Optional[GateDetection] = None
        self.latest_pose: Optional[GatePose] = None
        self.latest_image: Optional[np.ndarray] = None

    def _find_model_file(self, model_name: str) -> Optional[str]:
        """Search for model file in common locations.
        
        Searches for the model file in the following order:
        1. ./models/ (relative to current working directory)
        2. ../models/ (parent directory)
        3. models/ directory relative to this file's location
        4. models/ directory relative to autonomous-drone-racing package
        """
        # Common model directory names
        model_dirs = ['models', 'model']
        
        # Search paths relative to current working directory
        cwd = os.getcwd()
        for dir_name in model_dirs:
            for depth in [0, 1, 2]:
                path = os.path.join(cwd, '../' * depth, dir_name, model_name)
                abs_path = os.path.abspath(path)
                if os.path.exists(abs_path):
                    return abs_path
        
        # Search relative to this file's location
        this_file_dir = os.path.dirname(os.path.abspath(__file__))
        for dir_name in model_dirs:
            # Search in autonomous-drone-racing/models/
            path = os.path.join(this_file_dir, '..', '..', dir_name, model_name)
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                return abs_path
            # Search in auto_racing/models/
            path = os.path.join(this_file_dir, '..', '..', '..', dir_name, model_name)
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                return abs_path
        
        return None

    def _init_vision(self):
        """Initialize vision components."""
        # YOLO for segmentation
        model_path = None
        if self.config.yolo_model_name:
            model_path = self._find_model_file(self.config.yolo_model_name)
        
        if model_path:
            self.yolo = YOLO(model_path)
            print(f"Loaded YOLO model from: {model_path}")
        else:
            # Load default YOLOv8 segmentation model
            print(f"Model '{self.config.yolo_model_name}' not found, using default YOLOv8n-seg")
            self.yolo = YOLO("yolov8n-seg.pt")

        # Move to device
        # self.yolo.to(self.device)

        # QuAdGate for corner detection
        self.quad_gate = QuAdGate()
        self.gate_tracker = GateTracker(self.quad_gate)

        # Pose estimator
        self.pose_estimator = PoseEstimator(
            gate_width=self.config.gate_width,
            gate_height=self.config.gate_height,
            image_size=(self.config.image_width, self.config.image_height),
            camera_fov=self.config.camera_fov,
            camera_matrix=self.config.intrinsic_matrix,
            dist_coeffs=self.config.dist_coeffs,
        )

    def _init_state_estimation(
        self,
        known_gates: Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]] = None,
    ):
        """Initialize state estimation (EKF)."""
        self.ekf = ExtendedKalmanFilter(extrinsic_matrix=self.config.extrinsic_matrix)

        if known_gates:
            self.ekf.set_known_gates(known_gates)

        self.known_gates = known_gates or {}

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

        # Clear latest observations
        self.latest_detection = None
        self.latest_pose = None
        self.latest_image = None

    def _yolo_mask_to_single_channel(self, result) -> np.ndarray:
        """
        Convert YOLO segmentation output to a single channel mask.

        YOLO outputs multiple masks (one per detected object), this function
        merges them into a single binary mask.

        Args:
            result: YOLO inference result

        Returns:
            Single channel mask (H, W) with values in [0, 1]
        """
        if result.masks is None:
            return np.zeros((self.config.image_height, self.config.image_width), dtype=np.float32)

        # Get all masks from YOLO result
        masks = result.masks.data  # Shape: (num_masks, H, W)

        if masks.ndim == 2:
            masks = masks.unsqueeze(0)

        # Convert to numpy and merge all masks
        mask_np = masks.cpu().numpy()

        # Merge all object masks into one
        merged_mask = np.max(mask_np, axis=0)

        # Ensure binary mask
        merged_mask = (merged_mask > 0.5).astype(np.float32)

        return merged_mask

    def process_image(
        self,
        rgb_image: np.ndarray,
        current_time: float,
        save_mask_path: Optional[str] = None,
    ) -> Optional[GatePose]:
        """
        Process camera image through vision pipeline.

        Args:
            rgb_image: RGB image (H, W, 3) or (H, W, 4)
            current_time: Current simulation time
            save_mask_path: Optional path to save mask visualization

        Returns:
            Gate pose if detected, None otherwise
        """
        if self.config.convert_to_rgb:
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

        self.latest_image = rgb_image
        self.last_vision_time = current_time

        # Preprocess image
        # if rgb_image.shape[-1] == 4:
        #     rgb_image = rgb_image[..., :3]

        # Run YOLO inference
        # results = self.yolo.predict(rgb_image, verbose=False, device=self.device)
        results = self.yolo(
            source=rgb_image,
            conf=0.5,
            iou=0.7,
            max_det=10,
            half = True,
            rect = False,
            verbose=False,
        )

        # Get the first result
        result = results[0]

        # Convert YOLO output to single channel mask
        mask_np = self._yolo_mask_to_single_channel(result)

        # Detect corners
        detection = self.gate_tracker.update(mask_np)
        self.latest_detection = detection

        # Save mask visualization if path is provided
        if save_mask_path:
            if not self.config.convert_to_rgb:
                rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

            from ..vision.quad_gate import visualize_image_with_detection
            mask_vis = visualize_image_with_detection(rgb_image, mask_np, detection)
            cv2.imwrite(save_mask_path, mask_vis)

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

    def step(
        self,
        current_time: float,
        rgb_image: Optional[np.ndarray] = None,
        accel: Optional[np.ndarray] = None,
        gyro: Optional[np.ndarray] = None,
        dt: float = 0.002,  # Default 500 Hz
        save_mask_path: Optional[str] = None,
    ) -> Tuple[EKFState, Dict[str, Any]]:
        """
        Run one step of the pipeline.

        Args:
            current_time: Current simulation time
            rgb_image: Camera image (None if no new image available)
            accel: Accelerometer reading
            gyro: Gyroscope reading
            dt: Time step for EKF prediction
              > 原版代码根据当前时刻与上一次控制时刻的时间差获取dt，<br/>
              > 此处直接按500Hz设置dt
            save_mask_path: Optional path to save mask visualization

        Returns:
            Tuple of (state, info_dict)
        """
        self.total_steps += 1

        # Process vision if new image available
        gate_pose = None
        if rgb_image is not None:
            gate_pose = self.process_image(rgb_image, current_time, save_mask_path)
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

        return state, info

    def _advance_gate(self):
        """Advance to next gate."""
        self.gates_passed += 1
        self.current_gate_idx = (self.current_gate_idx + 1) % len(self.known_gates)

    def _quat_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion (w, x, y, z) to rotation matrix."""
        w, x, y, z = q / np.linalg.norm(q)
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )

    def get_state(self) -> EKFState:
        """Get current state estimate."""
        return self.ekf.state

    def set_gate_index(self, idx: int):
        """Manually set current gate index."""
        self.current_gate_idx = idx

    def load_model(self, yolo_model_path: str):
        """
        Load YOLO model. TODO: 未使用的方法

        Args:
            yolo_model_path: Path to YOLO checkpoint
        """
        self.yolo = YOLO(yolo_model_path)
        # self.yolo.to(self.device)


if __name__ == "__main__":
    # Test pipeline initialization
    print("Testing VisionRacingYOLOPipeline...")

    # Create known gates
    known_gates = {
        0: (np.array([2, 0, 1]), np.array([1, 0, 0, 0])),
        1: (np.array([4, 2, 1.2]), np.array([0.92, 0, 0, 0.38])),
        2: (np.array([4, 4, 1]), np.array([0.71, 0, 0, 0.71])),
    }

    config = YOLOPipelineConfig(
        vision_freq=24,
        device="cpu",
    )

    pipeline = VisionRacingYOLOPipeline(config=config, known_gates=known_gates)

    print(f"Pipeline initialized:")
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
    print("\nTesting state prediction steps...")
    for i in range(10):
        state, info = pipeline.step(
            current_time=i * 0.002,
            rgb_image=None,
        )
        if i % 5 == 0:
            print(f"  Step {i}: pos={info['position']}")

    # Test step with mock vision
    print("\nTesting with mock vision...")
    fake_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    state, info = pipeline.step(
        current_time=0.1,
        rgb_image=fake_image,
    )
    print(f"  Vision detected: {info['vision_detected']}")

    print(f"\nFinal state:")
    print(f"  Position: {pipeline.get_state().position}")
    print(f"  Velocity: {pipeline.get_state().velocity}")
    print(f"  Gates passed: {pipeline.gates_passed}")

    print("\nTest complete!")
