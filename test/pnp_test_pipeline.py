import numpy as np
from scipy.spatial.transform import Rotation
from loguru import logger

from src.state.ekf import ExtendedKalmanFilter
from src.vision.pose_estimator import PoseEstimator
from src.vision.quad_gate import GateDetection

# ==================== 已知参数 ====================
intrinsics = [233.4812810871969, 233.31138002134063, 228.4277126603732, 126.90905719756599]
intrinsic_matrix = np.array(
    [[intrinsics[0], 0.0, intrinsics[2]], [0.0, intrinsics[1], intrinsics[3]], [0.0, 0.0, 1.0]], dtype=np.float64
)
image_size = (512, 288)
h, w = image_size
extrinsic_matrix = np.array(
    [
        [-0.01571149301179632, -0.9988334570495578, -0.04566042121762354, -0.00036557132376473136],
        [0.3964618807115173, 0.03569953541070829, -0.9173568118862734, 0.061158889580899566],
        [0.9179167315884531, -0.03251566160929403, 0.39543849789832414, 0.03517857415040626],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
dist_coeffs = np.array([-0.27558041690741447, 0.08515382530472485, 0.0007373633751444942, 1.5895925597849783e-05])
points_3d = np.array(
    [
        [0, 1.35, 1.35],
        [0, -1.35, 1.35],
        [0, -1.35, -1.35],
        [0, 1.35, -1.35],
    ],
    dtype=np.float64,
)
points_2d = np.array(
    [
        [174, 123],
        [345, 110],
        [369, 276],
        [187, 297],
    ],
    dtype=np.float64,
)
body_pos_xyz_seu = np.array([2.787519131592388, 1.1825721482828748, 0.7859731579390311])
body_quat_xyzw_seu = np.array([0.020856436472970197, -0.08203751426503408, -0.687687877894269, 0.7210549479118359])
gate_pos_xyz_seu = np.array([-1.3686406697380282, 1.4187037807060632, 1.3489209174600654])
gate_quat_xyzw_seu = np.array([0.001287775970562029, 0.00275718150613899, -0.7267172459621076, 0.6869299702330955])

body_euler_seu = Rotation.from_quat(body_quat_xyzw_seu).as_euler("zyx", True)
gate_euler_seu = Rotation.from_quat(gate_quat_xyzw_seu).as_euler("zyx", True)
logger.debug(f"[已知条件] 机体位置: {body_pos_xyz_seu}")
logger.debug(f"[已知条件] 机体欧拉角: {body_euler_seu}")
logger.debug(f"[已知条件] 门框位置: {gate_pos_xyz_seu}")
logger.debug(f"[已知条件] 门框欧拉角: {gate_euler_seu}")

# region PoseEstimator
pose_estimator = PoseEstimator(
    gate_width=2.7,
    gate_height=2.7,
    camera_matrix=intrinsic_matrix,
    dist_coeffs=dist_coeffs,
    image_size=image_size,
)
gate_detection = GateDetection(
    corners=points_2d,
    confidence=0.9,
    is_complete=True,
    visible_corners=4,
)
gate_pose = pose_estimator.estimate_pose(gate_detection)
assert gate_pose is not None
logger.info(f"[PnP] 门框相对于相机的位置：{gate_pose.tvec.flatten()}")
logger.info(f"[PnP] 门框相对于相机的欧拉角：{Rotation.from_rotvec(gate_pose.rvec).as_euler('zyx', True)}")
# endregion PoseEstimator

# region EKF
ekf = ExtendedKalmanFilter(
    extrinsic_matrix=extrinsic_matrix,
)
ekf.set_known_gates({0: (gate_pos_xyz_seu, gate_quat_xyzw_seu)})
ekf.reset(
    position=body_pos_xyz_seu,
    velocity=np.zeros(3),
    orientation=body_quat_xyzw_seu,
)
gate_position_camera = gate_pose.position
gate_idx = 0
known_pos = ekf.known_gates[gate_idx]
R_cam_to_world = ekf.state.rotation_matrix @ ekf.R_cam_to_body
gate_pos_world = R_cam_to_world @ gate_position_camera + ekf.state.position
logger.info(f"[EKF] 门框在世界坐标系的位置：{gate_pos_world}")
# endregion EKF

logger.info(f"门框在世界坐标系的位置误差：{gate_pos_world - gate_pos_xyz_seu}")
logger.info(f"门框在世界坐标系的位置误差范数：{np.linalg.norm(gate_pos_world - gate_pos_xyz_seu)}")
