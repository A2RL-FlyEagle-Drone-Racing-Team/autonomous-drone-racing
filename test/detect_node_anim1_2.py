import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from scipy.spatial.transform import Rotation as Rot
import json
import glob


class OfflineDetector:
    def __init__(self, model_path, data_root, output_dir):
        self.model = YOLO(model_path)
        self.data_root = data_root
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 加载相机内参（来自 metadata.json）
        self.load_camera_params()
        # 加载门框世界位姿
        self.load_gate_pose()
        # 加载无人机真实位姿数据
        self.load_imu_data()

        # 3D参考点（门框角点，单位米）
        self.rectangle_3d_points = np.array(
            [
                [1.35, 1.35, 0],
                [-1.35, 1.35, 0],
                [-1.35, -1.35, 0],
                [1.35, -1.35, 0],
            ],
            dtype=np.float32,
        )

        # 相机系到机体系的旋转矩阵（固定）
        # 相机系: x右, y下, z前; 机体系: x前, y左, z上
        self.R_c_b = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float32)  # 将相机系点转到机体系

    def load_camera_params(self):
        meta_path = os.path.join(self.data_root, "metadata.json")
        with open(meta_path, "r") as f:
            meta = json.load(f)
        cam = meta["camera"]
        # 焦距（像素）由 sensor 尺寸和分辨率计算
        fx = cam["focal_length_mm"] * cam["resolution_x"] / cam["sensor_width_mm"]
        fy = cam["focal_length_mm"] * cam["resolution_y"] / cam["sensor_height_mm"]
        cx = cam["resolution_x"] / 2.0
        cy = cam["resolution_y"] / 2.0
        self.camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = np.zeros(5)  # 无畸变

    def load_gate_pose(self):
        meta_path = os.path.join(self.data_root, "metadata.json")
        with open(meta_path, "r") as f:
            meta = json.load(f)
        gate = meta["gate"]
        self.gate_pos_w = np.array(gate["position"], dtype=np.float32)  # 世界系位置
        euler_xyz = gate["euler_xyz_rad"]  # [roll, pitch, yaw]? 顺序xyz
        self.R_g_w = Rot.from_euler("xyz", euler_xyz).as_matrix()  # 门框系到世界系
        # self.R_g_w = self.R_g_w @ np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float32)

    def load_imu_data(self):
        csv_path = os.path.join(self.data_root, "imu_data.csv")
        self.imu_df = pd.read_csv(csv_path)
        # 提取位置和四元数（w,x,y,z）
        self.pos_w = self.imu_df[["pos_x", "pos_y", "pos_z"]].values.astype(np.float32)
        quat_xyzw = self.imu_df[["quat_x", "quat_y", "quat_z", "quat_w"]].values.astype(
            np.float32
        )  # scipy 需要 (x,y,z,w)
        self.quat_u = quat_xyzw  # 顺序已为 x,y,z,w

    def compute_ground_truth(self, frame_idx):
        """计算第 frame_idx 帧的门框相对于相机系的真实位姿"""
        # 无人机世界系位姿
        P_u_w = self.pos_w[frame_idx]
        q_u_xyzw = self.quat_u[frame_idx]
        R_u_w = Rot.from_quat(q_u_xyzw).as_matrix()  # 机体系 -> 世界系

        # 门框在相机系下的位置
        P_g_b = R_u_w.T @ (self.gate_pos_w - P_u_w)  # 在机体系
        P_g_c = self.R_c_b.T @ P_g_b  # 在相机系

        # 门框在相机系下的姿态（门框系 -> 相机系）
        R_g_b = R_u_w.T @ self.R_g_w
        R_g_c = self.R_c_b.T @ R_g_b

        # 将旋转矩阵转为欧拉角（zyx顺序，与原代码一致）
        r = Rot.from_matrix(R_g_c)
        euler_zyx = r.as_euler("zyx", degrees=True)  # [yaw, pitch, roll]?
        # 原代码中：euler = r.as_euler('zyx', degrees=True); roll, pitch, yaw = euler[2], euler[1], euler[0]
        roll = euler_zyx[2]
        pitch = euler_zyx[1]
        yaw = euler_zyx[0]

        return P_g_c, roll, pitch, yaw

    def process_frame(self, image_path):
        """对单帧图像进行推理，估计门框位姿"""
        img = cv2.imread(image_path)
        if img is None:
            return None, None, None, None

        # 可选：降低亮度（原代码有，但这里不做以保持简单）
        results = self.model(img, conf=0.25, iou=0.7, max_det=10, device="cpu", half=False, verbose=False)

        # 寻找矩形并计算最近距离
        best_pose = None
        best_distance = float("inf")
        best_center = None

        for result in results:
            if hasattr(result, "masks") and result.masks is not None:
                for mask in result.masks:
                    # 获取轮廓点（像素坐标）
                    points = mask.xy[0].astype(np.int32)
                    if len(points) < 4:
                        continue
                    # 近似多边形
                    epsilon = 0.02 * cv2.arcLength(points, True)
                    approx = cv2.approxPolyDP(points, epsilon, True)
                    if len(approx) != 4:
                        continue
                    # 整理角点顺序（与原代码一致：order_points）
                    ordered, center = self.order_points(approx.reshape(-1, 2))
                    # PnP求解
                    success, rvec, tvec = cv2.solvePnP(
                        self.rectangle_3d_points,
                        ordered.astype(np.float32),
                        self.camera_matrix,
                        self.dist_coeffs,
                        flags=cv2.SOLVEPNP_SQPNP,
                    )
                    if success:
                        dist = np.linalg.norm(tvec)
                        if dist < best_distance:
                            best_distance = dist
                            best_pose = (rvec, tvec)
                            best_center = center

        if best_pose is None:
            return None, None, None, None

        rvec, tvec = best_pose
        # 转换为欧拉角（与原代码一致）
        R, _ = cv2.Rodrigues(rvec)
        r = Rot.from_matrix(R)
        euler_zyx = r.as_euler("zyx", degrees=True)
        roll = euler_zyx[2]
        pitch = euler_zyx[1]
        yaw = euler_zyx[0]
        pos = tvec.flatten()  # 相机系位置

        return pos, roll, pitch, yaw

    def order_points(self, pts):
        """将四点排序为：右上、右下、左下、左上（与原代码一致）"""
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        rect[0] = pts[np.argmax(s)]  # 右下? 原代码是max sum -> 右下
        rect[1] = pts[np.argmax(diff)]  # 右上? diff最大 -> 右上
        rect[2] = pts[np.argmin(s)]  # 左上
        rect[3] = pts[np.argmin(diff)]  # 左下
        center = pts.sum(axis=0) / 4
        return rect, center

    def run(self):
        # 获取所有图像路径
        img_dir = os.path.join(self.data_root, "imgs")
        img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        n_frames = len(img_paths)
        print(f"Found {n_frames} images.")

        # 准备存储结果
        results = []
        for idx, img_path in enumerate(img_paths):
            print(f"Processing frame {idx:04d}...")
            # 估计值
            est_pos, est_roll, est_pitch, est_yaw = self.process_frame(img_path)
            # 真实值
            gt_pos, gt_roll, gt_pitch, gt_yaw = self.compute_ground_truth(idx)

            row = {
                "frame": idx,
                "est_x": est_pos[0] if est_pos is not None else np.nan,
                "est_y": est_pos[1] if est_pos is not None else np.nan,
                "est_z": est_pos[2] if est_pos is not None else np.nan,
                "est_roll": est_roll if est_roll is not None else np.nan,
                "est_pitch": est_pitch if est_pitch is not None else np.nan,
                "est_yaw": est_yaw if est_yaw is not None else np.nan,
                "gt_x": gt_pos[0],
                "gt_y": gt_pos[1],
                "gt_z": gt_pos[2],
                "gt_roll": gt_roll,
                "gt_pitch": gt_pitch,
                "gt_yaw": gt_yaw,
            }
            results.append(row)

        # 保存CSV
        df = pd.DataFrame(results)
        csv_path = os.path.join(self.output_dir, "results.csv")
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")

        # 绘图
        self.plot_results(df)

    def plot_results(self, df):
        # 位置曲线
        fig, axes = plt.subplots(2, 3, figsize=(12, 6))
        pos_labels = ["x", "y", "z"]
        for i, label in enumerate(pos_labels):
            ax = axes[0, i]
            ax.plot(df["frame"], df[f"est_{label}"], "b-", label="Estimate")
            ax.plot(df["frame"], df[f"gt_{label}"], "r--", label="Ground Truth")
            ax.set_title(f"Position {label}")
            ax.legend()
            ax.grid(True)

        # 姿态曲线
        att_labels = ["roll", "pitch", "yaw"]
        for i, label in enumerate(att_labels):
            ax = axes[1, i]
            ax.plot(df["frame"], df[f"est_{label}"], "b-", label="Estimate")
            ax.plot(df["frame"], df[f"gt_{label}"], "r--", label="Ground Truth")
            ax.set_title(f"Attitude {label} (deg)")
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, "comparison_plots.png")
        plt.savefig(plot_path)
        plt.show()
        print(f"Plots saved to {plot_path}")


if __name__ == "__main__":
    # 配置路径
    DATA_ROOT = r"D:\Documents\BIT\2025-2026_2\a2rl\monorace_perception\assets\anim_test-1"
    MODEL_PATH = "models/best.pt"
    OUTPUT_DIR = "output/detect_node_anim1_2"

    detector = OfflineDetector(MODEL_PATH, DATA_ROOT, OUTPUT_DIR)
    detector.run()
