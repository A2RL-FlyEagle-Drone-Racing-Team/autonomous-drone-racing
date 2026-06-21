"""
Gate Pose Estimator using Perspective-n-Point (PnP).

Estimates gate 6-DoF pose from 2D image corner detections
and known gate dimensions using cv2.solvePnP.
"""

import numpy as np
from typing import Optional, Tuple, NamedTuple
from dataclasses import dataclass
import cv2

from .quad_gate import GateDetection


@dataclass
class GatePose:
    """6-DoF pose of a gate in camera frame."""
    position: np.ndarray  # 3D position (x, y, z) in camera frame
    orientation: np.ndarray  # Rotation as quaternion (x, y, z, w)
    rotation_matrix: np.ndarray  # 3x3 rotation matrix
    rvec: np.ndarray  # Rodrigues rotation vector
    tvec: np.ndarray  # Translation vector
    confidence: float  # Pose estimation confidence
    reprojection_error: float  # Mean reprojection error in pixels
    distance: float  # Distance to gate center


class PoseEstimator:
    """
    Estimates gate pose from corner detections using PnP.

    Uses known gate dimensions and camera intrinsics to solve
    the Perspective-n-Point problem.
    """

    def __init__(
        self,
        gate_width: float = 1.0,
        gate_height: float = 1.0,
        camera_matrix: Optional[np.ndarray] = None,
        dist_coeffs: Optional[np.ndarray] = None,
        image_size: Tuple[int, int] = (64, 48),
        camera_fov: Optional[float] = 60.0,
        gate_points_3d: Optional[np.ndarray] = None,
    ):
        """
        Initialize pose estimator.

        Args:
            gate_width: Gate width in meters
            gate_height: Gate height in meters
            camera_matrix: 3x3 camera intrinsic matrix (computed if None)
            dist_coeffs: Distortion coefficients (zeros if None)
            image_size: (width, height) of image
            camera_fov: Camera field of view in degrees
        """
        self.gate_width = gate_width
        self.gate_height = gate_height
        self.image_size = image_size
        self.camera_fov = camera_fov

        # Compute camera matrix if not provided
        if camera_matrix is None:
            self.camera_matrix = self._compute_camera_matrix()
        else:
            self.camera_matrix = camera_matrix.astype(np.float64)

        # Default to zero distortion
        if dist_coeffs is None:
            self.dist_coeffs = np.zeros(5, dtype=np.float64)
        else:
            self.dist_coeffs = dist_coeffs.astype(np.float64)

        # 3D points of gate corners in gate frame
        # Order: TL, TR, BR, BL (matching QuAdGate output)
        # Gate center at origin, facing +Z direction
        # self.gate_points_3d = np.array([
        #     [-gate_width/2, -gate_height/2, 0],  # Top-left
        #     [gate_width/2, -gate_height/2, 0],   # Top-right
        #     [gate_width/2, gate_height/2, 0],    # Bottom-right
        #     [-gate_width/2, gate_height/2, 0],   # Bottom-left
        # ], dtype=np.float64)
        self.gate_points_3d = np.array([
            [0, gate_width/2, gate_height/2],
            [0, -gate_width/2, gate_height/2],
            [0, -gate_width/2, -gate_height/2],
            [0, gate_width/2, -gate_height/2],
        ], dtype=np.float64) if gate_points_3d is None else gate_points_3d

    def _compute_camera_matrix(self) -> np.ndarray:
        """Compute camera intrinsic matrix from FOV and image size."""
        width, height = self.image_size
        assert self.camera_fov is not None, "FOV or camera matrix must be provided"
        fov_rad = np.radians(self.camera_fov)

        # Focal length in pixels
        fx = width / (2 * np.tan(fov_rad / 2))
        fy = fx  # Square pixels

        # Principal point at image center
        cx = width / 2
        cy = height / 2

        return np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ], dtype=np.float64)

    def estimate_pose(
        self,
        detection: GateDetection,
        use_ransac: bool = True,
    ) -> Optional[GatePose]:
        """
        Estimate gate pose from corner detection.

        Args:
            detection: Gate detection with 4 corners
            use_ransac: Use RANSAC-based PnP for robustness

        Returns:
            GatePose or None if estimation fails
        """
        if detection is None or detection.corners is None:
            return None

        # Get image points (ensure float64)
        image_points = detection.corners.astype(np.float64)

        # Verify we have 4 points
        if len(image_points) != 4:
            return None

        try:    # 相机在门框局部坐标系下的位姿
            if use_ransac:
                # RANSAC-based PnP (more robust to outliers)
                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    self.gate_points_3d,
                    image_points,
                    self.camera_matrix,
                    self.dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
            else:
                # Standard PnP
                success, rvec, tvec = cv2.solvePnP(
                    self.gate_points_3d,
                    image_points,
                    self.camera_matrix,
                    self.dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
                inliers = np.arange(4)

            if not success:
                return None

            # Convert to numpy arrays
            rvec = rvec.flatten()
            tvec = tvec.flatten()

            # Compute rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rvec)

            # Convert to quaternion
            quaternion = self._rotation_matrix_to_quaternion(rotation_matrix)

            # Compute reprojection error
            projected_points, _ = cv2.projectPoints(
                self.gate_points_3d,
                rvec,
                tvec,
                self.camera_matrix,
                self.dist_coeffs,
            )
            projected_points = projected_points.reshape(-1, 2)
            reprojection_error = np.mean(
                np.linalg.norm(projected_points - image_points, axis=1)
            )

            # Compute distance to gate
            distance = np.linalg.norm(tvec)

            # Compute confidence based on reprojection error and detection confidence
            error_confidence = max(0, 1 - reprojection_error / 10)
            confidence = detection.confidence * error_confidence

            return GatePose(
                position=tvec,
                orientation=quaternion,
                rotation_matrix=rotation_matrix,
                rvec=rvec,
                tvec=tvec,
                confidence=confidence,
                reprojection_error=reprojection_error,
                distance=distance,  # type: ignore
            )

        except cv2.error:
            return None

    def _rotation_matrix_to_quaternion(self, R: np.ndarray) -> np.ndarray:
        """
        Convert 3x3 rotation matrix to quaternion (x, y, z, w).

        Args:
            R: 3x3 rotation matrix

        Returns:
            Quaternion as [x, y, z, w]
        """
        # Shepperd's method for numerical stability
        trace = np.trace(R)

        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

        q = np.array([x, y, z, w])
        return q / np.linalg.norm(q)  # Normalize

    def transform_to_world(
        self,
        gate_pose: GatePose,
        drone_position: np.ndarray,
        drone_orientation: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform gate pose from camera frame to world frame.

        Args:
            gate_pose: Gate pose in camera frame
            drone_position: Drone position in world frame
            drone_orientation: Drone orientation as quaternion (x, y, z, w)

        Returns:
            Tuple of (gate_position_world, gate_orientation_world)
        """
        # Convert drone orientation to rotation matrix
        R_drone = self._quaternion_to_rotation_matrix(drone_orientation)

        # Camera frame to drone frame transformation
        # Camera: z forward, x right, y down
        # Drone: x forward, y left, z up
        R_cam_to_drone = np.array([
            [0, 0, 1],
            [-1, 0, 0],
            [0, -1, 0],
        ])

        # Gate position in drone frame
        gate_pos_drone = R_cam_to_drone @ gate_pose.position

        # Gate position in world frame
        gate_pos_world = R_drone @ gate_pos_drone + drone_position

        # Gate orientation in world frame
        R_gate_cam = gate_pose.rotation_matrix
        R_gate_drone = R_cam_to_drone @ R_gate_cam
        R_gate_world = R_drone @ R_gate_drone

        gate_orientation_world = self._rotation_matrix_to_quaternion(R_gate_world)

        return gate_pos_world, gate_orientation_world

    def _quaternion_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        """
        Convert quaternion (x, y, z, w) to 3x3 rotation matrix.

        Args:
            q: Quaternion as [x, y, z, w]

        Returns:
            3x3 rotation matrix
        """
        x, y, z, w = q / np.linalg.norm(q)

        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
        ])

    def refine_pose(
        self,
        detection: GateDetection,
        initial_pose: GatePose,
        iterations: int = 10,
    ) -> GatePose:
        """
        Refine pose estimate using iterative optimization.

        Args:
            detection: Gate detection with corners
            initial_pose: Initial pose estimate
            iterations: Number of refinement iterations

        Returns:
            Refined GatePose
        """
        image_points = detection.corners.astype(np.float64)

        # Use initial pose as starting point
        rvec = initial_pose.rvec.copy()
        tvec = initial_pose.tvec.copy()

        # Iterative refinement using Levenberg-Marquardt
        rvec, tvec = cv2.solvePnPRefineLM(
            self.gate_points_3d,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            rvec.reshape(3, 1),
            tvec.reshape(3, 1),
        )

        rvec = rvec.flatten()
        tvec = tvec.flatten()

        # Recompute pose components
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        quaternion = self._rotation_matrix_to_quaternion(rotation_matrix)

        # Compute new reprojection error
        projected_points, _ = cv2.projectPoints(
            self.gate_points_3d,
            rvec,
            tvec,
            self.camera_matrix,
            self.dist_coeffs,
        )
        projected_points = projected_points.reshape(-1, 2)
        reprojection_error = np.mean(
            np.linalg.norm(projected_points - image_points, axis=1)
        )

        distance = np.linalg.norm(tvec)
        confidence = detection.confidence * max(0, 1 - reprojection_error / 10)

        return GatePose(
            position=tvec,
            orientation=quaternion,
            rotation_matrix=rotation_matrix,
            rvec=rvec,
            tvec=tvec,
            confidence=confidence,
            reprojection_error=reprojection_error,
            distance=distance,  # type: ignore
        )


class MultiGatePoseEstimator:
    """
    Handles pose estimation for multiple gates in a racing scenario.

    Tracks which gate is currently being approached and provides
    relative poses for navigation.
    """

    def __init__(
        self,
        gate_width: float = 1.0,
        gate_height: float = 1.0,
        **kwargs,
    ):
        """
        Initialize multi-gate estimator.

        Args:
            gate_width: Gate width in meters
            gate_height: Gate height in meters
            **kwargs: Additional args for PoseEstimator
        """
        self.estimator = PoseEstimator(
            gate_width=gate_width,
            gate_height=gate_height,
            **kwargs,
        )

        # Gate tracking
        self.current_gate_idx = 0
        self.gate_poses_history: list = []
        self.passed_gates: set = set()

    def update(
        self,
        detection: GateDetection,
        drone_position: np.ndarray,
        drone_orientation: np.ndarray,
    ) -> Optional[GatePose]:
        """
        Update pose estimation with new detection.

        Args:
            detection: Current gate detection
            drone_position: Drone world position
            drone_orientation: Drone world orientation

        Returns:
            Current gate pose or None
        """
        if detection is None:
            return None

        # Estimate pose in camera frame
        pose = self.estimator.estimate_pose(detection)
        if pose is None:
            return None

        # Store for history
        self.gate_poses_history.append({
            "gate_idx": self.current_gate_idx,
            "pose": pose,
            "drone_position": drone_position.copy(),
            "drone_orientation": drone_orientation.copy(),
        })

        # Keep history limited
        if len(self.gate_poses_history) > 100:
            self.gate_poses_history.pop(0)

        return pose

    def check_gate_passed(
        self,
        drone_position: np.ndarray,
        gate_position_world: np.ndarray,
        gate_normal: np.ndarray,
        tolerance: float = 0.5,
    ) -> bool:
        """
        Check if drone has passed through the current gate.

        Args:
            drone_position: Drone world position
            gate_position_world: Gate center in world frame
            gate_normal: Gate normal vector (direction through gate)
            tolerance: Distance tolerance for gate passage

        Returns:
            True if gate was passed
        """
        # Vector from gate to drone
        to_drone = drone_position - gate_position_world

        # Distance to gate plane
        plane_dist = np.dot(to_drone, gate_normal)

        # Check if we've passed through
        if plane_dist > tolerance and self.current_gate_idx not in self.passed_gates:
            self.passed_gates.add(self.current_gate_idx)
            return True

        return False

    def advance_gate(self, num_gates: int):
        """
        Advance to next gate in sequence.

        Args:
            num_gates: Total number of gates in track
        """
        self.current_gate_idx = (self.current_gate_idx + 1) % num_gates

    def reset(self):
        """Reset tracker state."""
        self.current_gate_idx = 0
        self.gate_poses_history.clear()
        self.passed_gates.clear()


if __name__ == "__main__":
    # Test pose estimation
    print("Testing PoseEstimator...")

    estimator = PoseEstimator(
        gate_width=1.0,
        gate_height=1.0,
        image_size=(64, 48),
        camera_fov=60.0,
    )

    print(f"Camera matrix:\n{estimator.camera_matrix}")

    # Simulate gate at 3m distance, slightly to the right
    gate_pos_true = np.array([0.5, 0.0, 3.0])  # x right, y down, z forward

    # Project gate corners to image
    R = np.eye(3)  # Gate facing camera
    rvec, _ = cv2.Rodrigues(R)
    tvec = gate_pos_true

    projected, _ = cv2.projectPoints(
        estimator.gate_points_3d,
        rvec,
        tvec,
        estimator.camera_matrix,
        estimator.dist_coeffs,
    )
    corners = projected.reshape(-1, 2)
    print(f"Projected corners:\n{corners}")

    # Create detection
    from .quad_gate import GateDetection
    detection = GateDetection(
        corners=corners.astype(np.float32),
        confidence=0.9,
        is_complete=True,
        visible_corners=4,
    )

    # Estimate pose
    pose = estimator.estimate_pose(detection)

    if pose is not None:
        print(f"\nEstimated position: {pose.position}")
        print(f"True position: {gate_pos_true}")
        print(f"Position error: {np.linalg.norm(pose.position - gate_pos_true):.4f} m")
        print(f"Reprojection error: {pose.reprojection_error:.4f} px")
        print(f"Distance: {pose.distance:.2f} m")
        print(f"Confidence: {pose.confidence:.3f}")
    else:
        print("Pose estimation failed!")

    print("\nTest complete!")
