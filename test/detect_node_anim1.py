import numpy as np
import cv2
from ultralytics import YOLO
from scipy.spatial.transform import Rotation as Rotat
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm

# 固定旋转：机体→相机（机体系：前左上，相机系：右下前）
# 机体系 X前 → 相机系 Z前（0,0,1）
# 机体系 Y左 → 相机系 X右的负方向（-1,0,0）
# 机体系 Z上 → 相机系 Y下的负方向（0,-1,0）
R_C_B = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float32)


class GateDetector:
    def __init__(self, model_path, camera_matrix, dist_coeffs, rect_3d_points):
        self.model = YOLO(model_path)
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.rect_3d_points = rect_3d_points.astype(np.float32)
        self.conf_thres = 0.25
        self.iou_thres = 0.7
        self.max_det = 10
        self.device = "cuda:0" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
        self.half = True if self.device == "cuda:0" else False

    def process_image(self, cv_image):
        # 亮度调整（与原ROS节点一致）
        ratio = 0.5
        brightness = -50
        img = cv2.addWeighted(cv_image, ratio, cv_image, 1 - ratio, brightness)

        results = self.model(
            img,
            conf=self.conf_thres,
            iou=self.iou_thres,
            max_det=self.max_det,
            device=self.device,
            half=self.half,
            rect=False,
            verbose=False,
        )

        best_distance = float("inf")
        best_tvec = None
        best_R = None

        if results and len(results) > 0:
            for result in results:
                if hasattr(result, "masks") and result.masks is not None:
                    for mask_tensor in result.masks:
                        mask_points = mask_tensor.xy[0].tolist()
                        if len(mask_points) < 4:
                            continue
                        mask_array = np.array([[int(p[0]), int(p[1])] for p in mask_points], dtype=np.int32)
                        epsilon = 0.02 * cv2.arcLength(mask_array, True)
                        approx = cv2.approxPolyDP(mask_array, epsilon, True)
                        if len(approx) == 4:
                            tvec, R, distance = self._process_rectangle(approx)
                            if tvec is not None and distance < best_distance:
                                best_distance = distance
                                best_tvec = tvec
                                best_R = R
        if best_tvec is not None:
            return best_tvec.flatten(), best_R
        return None, None

    def _process_rectangle(self, approx):
        points = np.array([point[0] for point in approx], dtype=np.float32)
        ordered_points, _ = self._order_points(points)

        success, rvec, tvec = cv2.solvePnP(
            self.rect_3d_points, ordered_points, self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_SQPNP
        )
        if success:
            R, _ = cv2.Rodrigues(rvec)
            distance = np.linalg.norm(tvec)
            return tvec.flatten(), R, distance
        return None, None, float("inf")

    # @staticmethod
    # def _order_points(pts):
    #     rect = np.zeros((4, 2), dtype=np.float32)
    #     s = pts.sum(axis=1)
    #     diff = np.diff(pts, axis=1)
    #     rect[0] = pts[np.argmax(s)]
    #     rect[1] = pts[np.argmax(diff)]
    #     rect[2] = pts[np.argmin(s)]
    #     rect[3] = pts[np.argmin(diff)]
    #     center = np.mean(pts, axis=0)
    #     return rect, center

    @staticmethod
    def _order_points(pts):
        # pts: (4,2) 浮点型
        center = np.mean(pts, axis=0)
        # 计算角度（图像坐标系：u右，v下）
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        # 转换为 [0, 2π)
        angles = (angles + 2 * np.pi) % (2 * np.pi)
        # 按角度升序（逆时针）
        idx = np.argsort(angles)
        sorted_pts = pts[idx]
        # 希望第一个点为右上角（角度约 315°）
        target = 315 * np.pi / 180
        start = np.argmin(np.abs(angles[idx] - target))
        ordered = np.roll(sorted_pts, -start, axis=0)
        return ordered, center


def compute_ground_truth(pos_cam, quat_wxyz):
    """
    根据无人机位姿（世界系ENU）计算机架在相机系下的位姿。
    输入：
        pos_cam : 无人机在世界系中的位置 [x,y,z] (ENU)
        quat_wxyz: 四元数 (w,x,y,z) 表示 机体→世界 的旋转
    输出：
        t_gate_C : 门框在相机系下的位置 (3,)
        R_gate_C : 门框在相机系下的旋转矩阵 (3,3)
        euler_deg: (roll, pitch, yaw) 单位度，顺序与估计一致
    """
    # 1. 机体→世界 旋转矩阵
    R_W_B = Rotat.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]).as_matrix()  # scipy顺序 [x,y,z,w]
    # 2. 世界→机体
    R_B_W = R_W_B.T
    # 3. 世界→相机
    R_C_W = R_C_B @ R_B_W

    # 4. 相机在世界系中的位置（假设相机与机体无平移）
    p_C_W = pos_cam
    # 5. 门框在世界系中的位置（来自metadata）
    p_G_W = np.array([0.0, 0.0, 0.0])
    # 6. 门框在相机系下的位置
    t_gate_C = R_C_W @ (p_G_W - p_C_W)

    # 7. 门框无旋转，姿态即为 R_C_W
    R_gate_C = R_C_W

    # 转换为欧拉角（zyx顺序，与估计相同）
    euler = Rotat.from_matrix(R_gate_C).as_euler("zyx", degrees=True)
    yaw, pitch, roll = euler[0], euler[1], euler[2]  # 原代码中 roll=euler[2], pitch=euler[1], yaw=euler[0]
    return t_gate_C, R_gate_C, (roll, pitch, yaw)


def load_imu_data(csv_path):
    df = pd.read_csv(csv_path)
    pos = df[["pos_x", "pos_y", "pos_z"]].values
    quat = df[["quat_w", "quat_x", "quat_y", "quat_z"]].values  # w,x,y,z
    return pos, quat


def load_images(image_dir, frame_range):
    images = []
    for i in range(frame_range[0], frame_range[1] + 1):
        img_path = os.path.join(image_dir, f"{i:04d}.jpg")
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        images.append(img)
    return images


def main():
    # 路径配置
    dataset_root = r"D:\Documents\BIT\2025-2026_2\a2rl\monorace_perception\assets\anim_test-1"
    img_dir = os.path.join(dataset_root, "imgs")
    csv_path = os.path.join(dataset_root, "imu_data.csv")
    metadata_path = os.path.join(dataset_root, "metadata.json")
    model_path = "models/best.pt"  # 请修改为实际路径

    # 输出目录
    output_dir = "./output/detect_node_anim1"
    os.makedirs(output_dir, exist_ok=True)

    # 读取相机内参
    with open(metadata_path, "r") as f:
        meta = json.load(f)
    cam = meta["camera"]
    fx = cam["focal_length_mm"] * cam["resolution_x"] / cam["sensor_width_mm"]
    fy = cam["focal_length_mm"] * cam["resolution_y"] / cam["sensor_height_mm"]
    cx = cam["resolution_x"] / 2.0
    cy = cam["resolution_y"] / 2.0
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros(5, dtype=np.float32)  # 无畸变

    # 门框3D点（与原代码一致）
    half_w = 1.35
    half_h = 1.35
    rect_3d = np.array(
        [
            [half_w, half_h, 0],
            [-half_w, half_h, 0],
            [-half_w, -half_h, 0],
            [half_w, -half_h, 0],
        ],
        dtype=np.float32,
    )

    detector = GateDetector(model_path, camera_matrix, dist_coeffs, rect_3d)

    # 加载数据
    images = load_images(img_dir, (0, 47))
    pos, quat = load_imu_data(csv_path)

    # 存储结果（每帧）
    results = []
    est_pos = []
    est_euler = []
    gt_pos = []
    gt_euler = []

    print("Processing frames...")
    for i, img in enumerate(tqdm(images)):
        # 估计
        tvec, R = detector.process_image(img)
        if tvec is not None:
            euler = Rotat.from_matrix(R).as_euler("zyx", degrees=True)
            roll, pitch, yaw = euler[2], euler[1], euler[0]
            est_pos.append(tvec)
            est_euler.append([roll, pitch, yaw])
        else:
            est_pos.append([np.nan, np.nan, np.nan])
            est_euler.append([np.nan, np.nan, np.nan])

        # 真实值
        pos_cam = pos[i]
        quat_wxyz = quat[i]
        t_gate_C, R_gate_C, euler_gt = compute_ground_truth(pos_cam, quat_wxyz)
        gt_pos.append(t_gate_C)
        gt_euler.append(euler_gt)

        # 保存该帧数据
        frame_data = {
            "frame": i,
            "gt_pos_x": t_gate_C[0],
            "gt_pos_y": t_gate_C[1],
            "gt_pos_z": t_gate_C[2],
            "gt_roll": euler_gt[0],
            "gt_pitch": euler_gt[1],
            "gt_yaw": euler_gt[2],
            "est_pos_x": tvec[0] if tvec is not None else np.nan,
            "est_pos_y": tvec[1] if tvec is not None else np.nan,
            "est_pos_z": tvec[2] if tvec is not None else np.nan,
            "est_roll": roll if tvec is not None else np.nan,
            "est_pitch": pitch if tvec is not None else np.nan,
            "est_yaw": yaw if tvec is not None else np.nan,
        }
        results.append(frame_data)

    # 保存CSV
    df_results = pd.DataFrame(results)
    csv_out = os.path.join(output_dir, "results.csv")
    df_results.to_csv(csv_out, index=False)
    print(f"Results saved to {csv_out}")

    # 绘图
    frames = np.arange(len(images))
    est_pos = np.array(est_pos)
    est_euler = np.array(est_euler)
    gt_pos = np.array(gt_pos)
    gt_euler = np.array(gt_euler)

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    titles_pos = ["X (m)", "Y (m)", "Z (m)"]
    titles_euler = ["Roll (deg)", "Pitch (deg)", "Yaw (deg)"]

    for idx in range(3):
        ax = axes[0, idx]
        ax.plot(frames, gt_pos[:, idx], "b-", label="Ground Truth")
        ax.plot(frames, est_pos[:, idx], "r--", label="Estimation")
        ax.set_title(titles_pos[idx])
        ax.legend()
        ax.grid(True)

    for idx in range(3):
        ax = axes[1, idx]
        ax.plot(frames, gt_euler[:, idx], "b-", label="Ground Truth")
        ax.plot(frames, est_euler[:, idx], "r--", label="Estimation")
        ax.set_title(titles_euler[idx])
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "pose_comparison.png")
    plt.savefig(plot_path, dpi=150)
    plt.show()
    print(f"Plot saved to {plot_path}")

    # 计算平均误差（有效帧）
    valid_mask = ~np.isnan(est_pos).any(axis=1)
    if np.any(valid_mask):
        pos_errors = np.linalg.norm(est_pos[valid_mask] - gt_pos[valid_mask], axis=1)
        print(f"Average position error: {np.mean(pos_errors):.4f} m")
        valid_euler = ~np.isnan(est_euler).any(axis=1)
        euler_errors = np.linalg.norm(est_euler[valid_euler] - gt_euler[valid_euler], axis=1)
        print(f"Average Euler angle error: {np.mean(euler_errors):.4f} deg")


if __name__ == "__main__":
    main()
