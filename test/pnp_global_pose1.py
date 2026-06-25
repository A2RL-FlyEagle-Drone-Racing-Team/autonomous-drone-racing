import os
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from ultralytics import YOLO

# --------------------- 配置 ---------------------
DATASET_DIR = r"D:\Documents\BIT\2025-2026_2\a2rl\monorace_perception\assets\pose_test-1"
MODEL_PATH = r"models/best.pt"  # 新分割模型
OUTPUT_DIR = "output/pnp_global_pose1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 相机固定旋转：机体系 -> 相机系 (右下前)
R_cam_body = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float32)

# 门框物理尺寸 (宽高均为2.7m)
HALF = 1.35

# 门框3D点 (门框系: x前(法线), y左, z上) 顺序: 左上,右上,右下,左下
world_pts_3d = np.array(
    [
        [0, HALF, HALF],
        [0, -HALF, HALF],
        [0, -HALF, -HALF],
        [0, HALF, -HALF],
    ],  # 左上  # 右上  # 右下  # 左下
    dtype=np.float32,
)

# --------------------- 读取元数据 ---------------------
with open(os.path.join(DATASET_DIR, "metadata.json"), "r") as f:
    meta = json.load(f)

cam = meta["camera"]
fx = cam["focal_length_mm"] / cam["sensor_width_mm"] * cam["resolution_x"]
fy = cam["focal_length_mm"] / cam["sensor_height_mm"] * cam["resolution_y"]
cx = cam["resolution_x"] / 2
cy = cam["resolution_y"] / 2
camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros(5, dtype=np.float32)  # 假设无畸变

# --------------------- 读取IMU真值 ---------------------
df_gt = pd.read_csv(os.path.join(DATASET_DIR, "imu_data.csv"))
frames = df_gt["frame"].values
times = df_gt["time"].values
pos_gt = df_gt[["pos_x", "pos_y", "pos_z"]].values  # (N,3)
quat_gt = df_gt[["quat_x", "quat_y", "quat_z", "quat_w"]].values  # (N,4) xyzw

# --------------------- 加载YOLO模型 ---------------------
model = YOLO(MODEL_PATH)


# --------------------- 辅助函数 ---------------------
def order_points(pts):
    """将四个点按顺时针顺序排列，从左上角开始"""
    # 计算中心
    center = np.mean(pts, axis=0)
    # 计算角度 (atan2(y-cy, x-cx)), 范围[-pi, pi], 转为[0, 2pi)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    angles = np.where(angles < 0, angles + 2 * np.pi, angles)
    # 按角度排序 (顺时针，因为图像y向下)
    idx = np.argsort(angles)
    pts_sorted = pts[idx]
    # 找到左上角 (v最小，若并列取u最小)
    top_idx = np.argmin(pts_sorted[:, 1])
    # 循环移位使左上角在第一位
    pts_shifted = np.roll(pts_sorted, -top_idx, axis=0)
    # 确保顺时针顺序: 检查第二个点是否为右上 (u较大)
    if pts_shifted[1, 0] < pts_shifted[3, 0]:
        # 若第二个点u较小，则可能是逆时针，反转
        pts_shifted = pts_shifted[::-1]
        # 重新调整左上角为第一位
        top_idx = np.argmin(pts_shifted[:, 1])
        pts_shifted = np.roll(pts_shifted, -top_idx, axis=0)
    return pts_shifted, center


def solve_pnp(points_2d):
    """求解PnP，返回旋转矩阵和平移向量"""
    success, rvec, tvec = cv2.solvePnP(
        world_pts_3d, points_2d.astype(np.float32), camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_SQPNP
    )
    if not success:
        return None, None
    R_world_cam, _ = cv2.Rodrigues(rvec)
    return R_world_cam, tvec.flatten()


# --------------------- 存储结果 ---------------------
results = []
for idx, frame in enumerate(frames):
    # 读取图像
    img_path = os.path.join(DATASET_DIR, f"imgs/{frame:04d}.jpg")
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: image {img_path} not found, skip.")
        continue

    # YOLO推理
    img = cv2.resize(img, (640, 640))
    res = model(
        img,
        conf=0.25,
        iou=0.7,
        max_det=10,
        device="cuda:0",
        half=True,
        verbose=False,
    )
    if len(res) == 0 or res[0].masks is None:
        print(f"Frame {frame}: no mask detected.")
        # 记录NaN
        row = [frame, times[idx]] + [np.nan] * 27  # 占位
        results.append(row)
        continue

    # 取第一个mask（通常只有一个门框）
    mask = res[0].masks[0]
    points = mask.xy[0]  # (N,2) 原始图像坐标
    if len(points) < 4:
        print(f"Frame {frame}: mask has insufficient points.")
        row = [frame, times[idx]] + [np.nan] * 27
        results.append(row)
        continue

    # 近似四边形
    approx = cv2.approxPolyDP(
        points.astype(np.float32), epsilon=0.02 * cv2.arcLength(points.astype(np.float32), True), closed=True
    )
    if len(approx) != 4:
        print(f"Frame {frame}: approximated polygon has {len(approx)} vertices, skip.")
        row = [frame, times[idx]] + [np.nan] * 27
        results.append(row)
        continue

    # 获取四个顶点并排序 (顺时针，从左上开始)
    pts_2d = np.array([p[0] for p in approx], dtype=np.float32)
    pts_ordered, _ = order_points(pts_2d)

    # PnP求解
    R_world_cam_est, t_cam_world_est = solve_pnp(pts_ordered)
    if R_world_cam_est is None or t_cam_world_est is None:
        print(f"Frame {frame}: PnP failed.")
        row = [frame, times[idx]] + [np.nan] * 27
        results.append(row)
        continue

    # ---------- 估计值 ----------
    # 无人机位置估计 (地面系)
    p_drone_est = -R_world_cam_est.T @ t_cam_world_est
    # 无人机姿态估计 (地面系)
    R_drone_est = R_world_cam_est
    euler_est = R.from_matrix(R_drone_est).as_euler("zyx", degrees=True)  # [yaw, pitch, roll]

    # ---------- 真值 ----------
    p_drone_true = pos_gt[idx]
    euler_true = R.from_quat(quat_gt[idx]).as_euler("zyx", degrees=True)  # [yaw, pitch, roll]

    # 收集数据
    row = [
        frame,
        times[idx],
        p_drone_est[0],
        p_drone_est[1],
        p_drone_est[2],
        euler_est[0],
        euler_est[1],
        euler_est[2],
        p_drone_true[0],
        p_drone_true[1],
        p_drone_true[2],
        euler_true[0],
        euler_true[1],
        euler_true[2],
    ]
    results.append(row)

# --------------------- 保存CSV ---------------------
columns = [
    "frame",
    "time",
    "drone_pos_est_x",
    "drone_pos_est_y",
    "drone_pos_est_z",
    "drone_att_est_yaw",
    "drone_att_est_pitch",
    "drone_att_est_roll",
    "drone_pos_true_x",
    "drone_pos_true_y",
    "drone_pos_true_z",
    "drone_att_true_yaw",
    "drone_att_true_pitch",
    "drone_att_true_roll",
]
df = pd.DataFrame(results, columns=columns)
csv_path = os.path.join(OUTPUT_DIR, "pose_estimation.csv")
df.to_csv(csv_path, index=False)


# --------------------- 绘图 ---------------------
def plot_comparison(df, save_path, title, var_names, est_cols, true_cols, xlabel="Frame"):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    for i, (name, est_col, true_col) in enumerate(zip(var_names, est_cols, true_cols)):
        ax = axes[i]
        ax.plot(df["frame"], df[est_col], "o-", label="Estimate", markersize=4)
        ax.plot(df["frame"], df[true_col], "s-", label="True", markersize=4)
        ax.set_title(name)
        ax.set_xlabel(xlabel)
        ax.legend()
        ax.grid()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# 无人机位置和姿态
var_names = ["Pos X", "Pos Y", "Pos Z", "Yaw", "Pitch", "Roll"]
est_cols = [
    "drone_pos_est_x",
    "drone_pos_est_y",
    "drone_pos_est_z",
    "drone_att_est_yaw",
    "drone_att_est_pitch",
    "drone_att_est_roll",
]
true_cols = [
    "drone_pos_true_x",
    "drone_pos_true_y",
    "drone_pos_true_z",
    "drone_att_true_yaw",
    "drone_att_true_pitch",
    "drone_att_true_roll",
]
plot_comparison(
    df,
    os.path.join(OUTPUT_DIR, "drone_comparison.png"),
    "Drone Pose vs Truth",
    var_names,
    est_cols,
    true_cols,
)
print(f"Results saved to {OUTPUT_DIR}")
