"""
quad_gate & pose_estimator 测试
"""

import cv2
import numpy as np

from src.vision.pose_estimator import PoseEstimator
from src.vision.quad_gate import GateDetection, QuAdGate

mask_path = r"D:\Documents\BIT\2025-2026_2\a2rl\monorace_perception\assets\binary_test-1.jpg"
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
assert mask is not None

gate_width = 4.0 # m
gate_height = 4.0 # m
image_size = (128, 128)
cam_size = 36 # mm
cam_focal_length = 75 # mm
cam_fov_deg = 2 * np.atan2(cam_size / 2, cam_focal_length) * 180 / np.pi # deg
corners = np.array([[34, 26], [87, 33], [85, 91], [36, 80]])

detector = QuAdGate()

# Detect
detection = detector.detect(mask)

if detection is not None:
    print(f"Detection confidence: {detection.confidence:.3f}")
    print(f"Complete: {detection.is_complete}")
    print(f"Visible corners: {detection.visible_corners}")
    print(f"Detected corners:\n{detection.corners}")
    print(f"Original corners:\n{corners}")

    # Compute corner error
    error = np.mean(np.abs(detection.corners - corners))
    print(f"Mean corner error: {error:.2f} pixels")
else:
    print("No detection!")

# Estimate
estimator = PoseEstimator(
    gate_width=gate_width,
    gate_height=gate_height,
    image_size=image_size,
    camera_fov=cam_fov_deg,
)

print(f"Camera matrix:\n{estimator.camera_matrix}")

cam_euler = np.deg2rad(np.array([0, 20, -30]))  # xyz
gate_euler = np.array([0, 0, 0])
gate_rvec = cam_euler - gate_euler

gate_pos = np.array([0, 0, 0], dtype=np.float32)  # xyz, m
cam_pos = np.array([-15, -9, 6], dtype=np.float32)  # xyz, m
gate_tvec = cam_pos - gate_pos

projected_corners = cv2.projectPoints(
    estimator.gate_points_3d,
    gate_rvec,
    gate_tvec,
    estimator.camera_matrix,
    estimator.dist_coeffs,
)[0].reshape(-1, 2)

print(f"Projected corners:\n{projected_corners}")

detection = GateDetection(
    corners=projected_corners.astype(np.float32),
    confidence=0.9,
    is_complete=True,
    visible_corners=4,
)
# Estimate pose
pose = estimator.estimate_pose(detection)
if pose is not None:
    print(f"\nEstimated position: {pose.position}")
    print(f"True position: {gate_tvec}")
    print(f"Position error: {np.linalg.norm(pose.position - gate_tvec):.4f} m")
    print(f"Reprojection error: {pose.reprojection_error:.4f} px")
    print(f"Distance: {pose.distance:.2f} m")
    print(f"Confidence: {pose.confidence:.3f}")
else:
    print("Pose estimation failed!")

print("\nTest complete!")
