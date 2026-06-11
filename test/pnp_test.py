import numpy as np
import cv2
from scipy.spatial.transform import Rotation

# ==================== 已知参数 ====================
intrinsics = [233.4812810871969, 233.31138002134063, 228.4277126603732, 126.90905719756599]
h, w = 512, 288
extrinsic_matrix = np.array([
    [-0.01571149301179632, -0.9988334570495578, -0.04566042121762354, -0.00036557132376473136],
    [0.3964618807115173, 0.03569953541070829, -0.9173568118862734, 0.061158889580899566],
    [0.9179167315884531, -0.03251566160929403, 0.39543849789832414, 0.03517857415040626],
    [0.0, 0.0, 0.0, 1.0]
])
dist_coeffs = np.array([-0.27558041690741447, 0.08515382530472485, 0.0007373633751444942, 1.5895925597849783e-05])
points_3d = np.array([
    [0, 1.35, 1.35],
    [0, -1.35, 1.35],
    [0, -1.35, -1.35],
    [0, 1.35, -1.35]
], dtype=np.float64)
points_2d = np.array([
    [174, 123],
    [345, 110],
    [369, 276],
    [187, 297]
], dtype=np.float32)
body_pos_xyz_seu = np.array([2.787519131592388, 1.1825721482828748, 0.7859731579390311])
body_quat_xyzw_seu = np.array([0.020856436472970197, -0.08203751426503408, -0.687687877894269, 0.7210549479118359])
gate_pos_xyz_seu = np.array([-1.3686406697380282, 1.4187037807060632, 1.3489209174600654])
gate_quat_xyzw_seu = np.array([0.001287775970562029, 0.00275718150613899, -0.7267172459621076, 0.6869299702330955])

# ==================== 1. 内参矩阵 ====================
fx, fy, cx, cy = intrinsics
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]], dtype=np.float64)

# ==================== 辅助函数 ====================
def build_transform(translation, quat_xyzw):
    """从平移向量和四元数（xyzw）构造 4x4 变换矩阵（局部→世界）"""
    R = Rotation.from_quat(quat_xyzw).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = translation
    return T

def invert_transform(T):
    """求 4x4 变换矩阵的逆（假设是刚体变换）"""
    R_inv = T[:3, :3].T
    t_inv = -R_inv @ T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    return T_inv

def seu_to_enu_point(point_seu):
    """将南-东-天（SEU）坐标系下的点转换到东-北-天（ENU）"""
    x_s, y_e, z_u = point_seu
    return np.array([y_e, -x_s, z_u])

def seu_to_enu_quat(quat_seu_xyzw):
    """将 SEU 坐标系下的四元数转换到 ENU 坐标系"""
    # SEU → ENU 的旋转矩阵 (将向量从 SEU 转换到 ENU)
    R_seu2enu = np.array([[0, 1, 0],
                          [-1, 0, 0],
                          [0, 0, 1]])
    # 获取 SEU 下的旋转矩阵
    R_seu = Rotation.from_quat(quat_seu_xyzw).as_matrix()
    # 在 ENU 下表示同一旋转
    R_enu = R_seu2enu @ R_seu @ R_seu2enu.T
    quat_enu = Rotation.from_matrix(R_enu).as_quat()  # 返回 xyzw
    return quat_enu

# ==================== 2. 坐标系转换 ====================
# 2.1 机体 → SEU 变换矩阵
T_body2seu = build_transform(body_pos_xyz_seu, body_quat_xyzw_seu)

# 2.2 相机 → 机体 变换矩阵（直接使用给定的 extrinsic_matrix）
T_cam2body = extrinsic_matrix

# 2.3 相机 → SEU 变换矩阵
T_cam2seu = T_body2seu @ T_cam2body
cam_pos_seu = T_cam2seu[:3, 3]
cam_rot_seu = T_cam2seu[:3, :3]
cam_quat_seu = Rotation.from_matrix(cam_rot_seu).as_quat()

# 2.4 将相机位姿转换到 ENU
cam_pos_enu = seu_to_enu_point(cam_pos_seu)
cam_quat_enu = seu_to_enu_quat(cam_quat_seu)

# 2.5 门框真实位姿（SEU → ENU）
gate_pos_enu_true = seu_to_enu_point(gate_pos_xyz_seu)
gate_quat_enu_true = seu_to_enu_quat(gate_quat_xyzw_seu)

print("相机在 ENU 下的真实位置 (x_e, y_n, z_u):", cam_pos_enu)
print("相机在 ENU 下的真实四元数 (xyzw):", cam_quat_enu)
print("门框在 ENU 下的真实位置:", gate_pos_enu_true)
print("门框在 ENU 下的真实四元数:", gate_quat_enu_true)

# ==================== 3. PnP 求解 ====================
# 使用 solvePnPRansac 获取门框坐标系 → 相机坐标系的变换
success, rvec, tvec, inliers = cv2.solvePnPRansac(
    points_3d, points_2d, K, dist_coeffs,
    iterationsCount=100,
    reprojectionError=8.0,
    confidence=0.99,
    flags=cv2.SOLVEPNP_ITERATIVE
)
if not success:
    print("PnP 求解失败")
    exit(1)

# 将旋转向量转换为旋转矩阵
R_gate2cam, _ = cv2.Rodrigues(rvec)
t_gate2cam = tvec.flatten()

# 构建门框 → 相机 变换矩阵
T_gate2cam = np.eye(4)
T_gate2cam[:3, :3] = R_gate2cam
T_gate2cam[:3, 3] = t_gate2cam

print("\nPnP 求解完成，内点数量:", len(inliers) if inliers is not None else 0)
print("门框→相机 平移向量 (相机系):", t_gate2cam)

# ==================== 4. 利用 PnP 结果和真实相机位姿求门框位姿 ====================
# 4.1 相机 → ENU 变换矩阵（真实值）
T_cam2enu = build_transform(cam_pos_enu, cam_quat_enu)

# 4.2 门框 → ENU 变换矩阵（估计值）
T_gate2enu_est = T_cam2enu @ T_gate2cam
gate_pos_enu_est = T_gate2enu_est[:3, 3]
gate_rot_enu_est = T_gate2enu_est[:3, :3]
gate_quat_enu_est = Rotation.from_matrix(gate_rot_enu_est).as_quat()

# 4.3 门框 → 机体 变换矩阵
T_gate2body = T_cam2body @ T_gate2cam
gate_pos_body = T_gate2body[:3, 3]
gate_rot_body = T_gate2body[:3, :3]
gate_quat_body = Rotation.from_matrix(gate_rot_body).as_quat()
# 将机体坐标系下的旋转矩阵转换为欧拉角（按 ZYX 顺序，即 yaw, pitch, roll）
# 这里使用 scipy 的 from_matrix 然后 as_euler('zyx')，得到弧度
euler_body = Rotation.from_matrix(gate_rot_body).as_euler('zyx')
# 为便于理解，转换为度并注明顺序
euler_body_deg = np.rad2deg(euler_body)

print("\n=== 门框在 ENU 下的估计结果 ===")
print("估计位置 (x_e, y_n, z_u):", gate_pos_enu_est)
print("估计四元数 (xyzw):", gate_quat_enu_est)

print("\n=== 门框在机体坐标系下的结果 ===")
print("门框中心相对于机体的坐标 (前, 左, 上):", gate_pos_body)
print("门框姿态四元数 (xyzw) 相对于机体:", gate_quat_body)
print("门框姿态欧拉角 (ZYX: yaw, pitch, roll) 弧度:", euler_body)
print("门框姿态欧拉角 (ZYX: yaw, pitch, roll) 度:", euler_body_deg)

# ==================== 与真实值比较 ====================
# 位置误差
pos_error = np.linalg.norm(gate_pos_enu_est - gate_pos_enu_true)
# 姿态误差：计算相对旋转的角度差
R_true = Rotation.from_quat(gate_quat_enu_true).as_matrix()
R_est = gate_rot_enu_est
R_rel = R_est @ R_true.T  # 估计 * 真实逆
# 相对旋转的角度（弧度）
angle_error_rad = np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1.0, 1.0))
angle_error_deg = np.rad2deg(angle_error_rad)

print("\n=== 与真实门框位姿比较 (ENU) ===")
print("位置误差 (米):", pos_error)
print("姿态角误差 (度):", angle_error_deg)