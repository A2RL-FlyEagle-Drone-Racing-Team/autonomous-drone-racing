import numpy as np
import cv2
from scipy.spatial.transform import Rotation

# ==================== 已知参数 ====================
"""
坐标轴定义
* 地面系：南东天
* 机体系：前左上
* 相机系：右下前
* 图像系：右下
"""
# 相机内参
intrinsics = [233.4812810871969, 233.31138002134063, 228.4277126603732, 126.90905719756599]
# 图像高和宽（像素）
h, w = 512, 288
# 相机外参矩阵（机体系和相机系之间的变换）
extrinsic_matrix = np.array([
    [-0.01571149301179632, -0.9988334570495578, -0.04566042121762354, -0.00036557132376473136],
    [0.3964618807115173, 0.03569953541070829, -0.9173568118862734, 0.061158889580899566],
    [0.9179167315884531, -0.03251566160929403, 0.39543849789832414, 0.03517857415040626],
    [0.0, 0.0, 0.0, 1.0]
])
# 畸变参数
dist_coeffs = np.array([-0.27558041690741447, 0.08515382530472485, 0.0007373633751444942, 1.5895925597849783e-05])
# 门框角点相对于门框中心坐标（米）
points_3d = np.array([
    [0, 1.35, 1.35],
    [0, -1.35, 1.35],
    [0, -1.35, -1.35],
    [0, 1.35, -1.35]
], dtype=np.float64)
# 门框角点在图像中投影的坐标（像素）
points_2d = np.array([
    [174, 123],
    [345, 110],
    [369, 276],
    [187, 297],
], dtype=np.float32)
# 机体世界系坐标（南东天，米）
body_pos_xyz_seu = np.array([2.787519131592388, 1.1825721482828748, 0.7859731579390311])
# 机体世界系姿态四元数（南东天，米）
body_quat_xyzw_seu = np.array([0.020856436472970197, -0.08203751426503408, -0.687687877894269, 0.7210549479118359])
# 门框世界系坐标（南东天，米）
gate_pos_xyz_seu = np.array([-1.3686406697380282, 1.4187037807060632, 1.3489209174600654])
# 门框世界系姿态四元数（南东天，米）
gate_quat_xyzw_seu = np.array([0.001287775970562029, 0.00275718150613899, -0.7267172459621076, 0.6869299702330955])
# [-93.22465952,   0.10979501,   0.33097762]

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

print("相机在 SEU 下的真实位置 (x_s, y_e, z_u):", cam_pos_seu)
print("相机在 SEU 下的真实四元数 (xyzw):", cam_quat_seu)
print("门框在 SEU 下的真实位置（x_s, y_e, z_u）:", gate_pos_xyz_seu)
print("门框在 SEU 下的真实四元数（xyzw）:", gate_quat_xyzw_seu)

# ==================== 3. PnP 求解 ====================
# 使用 solvePnPRansac 获取门框坐标系 → 相机坐标系的变换
success, rvec, tvec, inliers = cv2.solvePnPRansac(
    points_3d, points_2d, K, None,
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
# 4.1 相机 → SEU 变换矩阵（真实值）
T_cam2seu = build_transform(cam_pos_seu, cam_quat_seu)

# 4.2 门框 → SEU 变换矩阵（估计值）
T_gate2seu_est = T_cam2seu @ T_gate2cam
gate_pos_seu_est = T_gate2seu_est[:3, 3]
gate_rot_seu_est = T_gate2seu_est[:3, :3]
gate_quat_seu_est = Rotation.from_matrix(gate_rot_seu_est).as_quat()

# 4.3 门框 → 机体 变换矩阵
T_gate2body = T_cam2body @ T_gate2cam
gate_pos_body = T_gate2body[:3, 3]
gate_rot_body = T_gate2body[:3, :3]
gate_quat_body = Rotation.from_matrix(gate_rot_body).as_quat()
# 将机体坐标系下的旋转矩阵转换为欧拉角（按 ZYX 顺序，即 yaw, pitch, roll）
euler_body = Rotation.from_matrix(gate_rot_body).as_euler('zyx')
euler_body_deg = np.rad2deg(euler_body)

print("\n=== 门框在 SEU 下的估计结果 ===")
print("估计位置 (x_s, y_e, z_u):", gate_pos_seu_est)
print("估计四元数 (xyzw):", gate_quat_seu_est)

print("\n=== 门框在机体坐标系下的结果 ===")
print("门框中心相对于机体的坐标 (前, 左, 上):", gate_pos_body)
print("门框姿态四元数 (xyzw) 相对于机体:", gate_quat_body)
print("门框姿态欧拉角 (ZYX: yaw, pitch, roll) 弧度:", euler_body)
print("门框姿态欧拉角 (ZYX: yaw, pitch, roll) 度:", euler_body_deg)

# ==================== 与真实值比较 ====================
# 位置误差（SEU 坐标系下）
pos_error = np.linalg.norm(gate_pos_seu_est - gate_pos_xyz_seu)
# 姿态误差：计算相对旋转的角度差
R_true = Rotation.from_quat(gate_quat_xyzw_seu).as_matrix()
R_est = gate_rot_seu_est
R_rel = R_est @ R_true.T
angle_error_rad = np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1.0, 1.0))
angle_error_deg = np.rad2deg(angle_error_rad)

print("\n=== 与真实门框位姿比较 (SEU) ===")
print("位置误差 (米):", pos_error)
print("姿态角误差 (度):", angle_error_deg)



# 已知真实门框在 SEU 下的位姿
T_gate2seu_true = build_transform(gate_pos_xyz_seu, gate_quat_xyzw_seu)
# 相机在 SEU 下的真实位姿（已计算 cam_pos_seu, cam_quat_seu）
T_cam2seu_true = build_transform(cam_pos_seu, cam_quat_seu)
# 门框 → 相机 变换（真实值）
T_gate2cam_true = invert_transform(T_cam2seu_true) @ T_gate2seu_true

# 将 points_3d 转换到相机坐标系，再投影
points_cam = (T_gate2cam_true[:3,:3] @ points_3d.T).T + T_gate2cam_true[:3,3]
points_proj = (K @ points_cam.T).T
points_proj = points_proj[:, :2] / points_proj[:, 2:]

# 比较 points_proj 与 points_2d 的顺序相似度
print(points_proj)