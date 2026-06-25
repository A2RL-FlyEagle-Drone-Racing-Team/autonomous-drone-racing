# region 读取数据集（不变）
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.spatial.transform import Rotation

# 新增 YOLO 导入
from ultralytics import YOLO

# 不再需要 pipeline 和 pose_estimator
# from src.pipeline.vision_pipeline_yolo import VisionRacingYOLOPipeline, YOLOPipelineConfig
# from src.vision.pose_estimator import PoseEstimator


@dataclass
class Camera:
    focal_length_mm: float
    sensor_width_mm: float
    sensor_height_mm: float
    resolution_x: int
    resolution_y: int
    pixel_aspect_x: float
    pixel_aspect_y: float

    @classmethod
    def from_dict(cls, data: dict) -> "Camera":
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def camera_matrix(self):
        return np.array(
            [
                [self.focal_length_mm * self.sensor_width_mm / self.pixel_aspect_x, 0, self.resolution_x / 2],
                [0, self.focal_length_mm * self.sensor_height_mm / self.pixel_aspect_y, self.resolution_y / 2],
                [0, 0, 1],
            ]
        )


@dataclass
class Gate:
    position: List[float]
    euler_xyz_rad: List[float]
    dimensions: List[float]

    @classmethod
    def from_dict(cls, data: dict) -> "Gate":
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def gate_points_3d(self):
        return np.array(
            [
                [0, self.dimensions[1] / 2, self.dimensions[2] / 2],
                [0, -self.dimensions[1] / 2, self.dimensions[2] / 2],
                [0, -self.dimensions[1] / 2, -self.dimensions[2] / 2],
                [0, self.dimensions[1] / 2, -self.dimensions[2] / 2],
            ],
            dtype=np.float32,
        )


@dataclass
class SceneConfig:
    fps: int
    frame_range: List[int]
    camera: Camera
    gate: Gate
    dataset_dir: Path

    @classmethod
    def from_dict(cls, file_path: Path, data: dict) -> "SceneConfig":
        return cls(
            fps=data["fps"],
            frame_range=data["frame_range"],
            camera=Camera.from_dict(data["camera"]),
            gate=Gate.from_dict(data["gate"]),
            dataset_dir=file_path.parent,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def load_imu_data(self) -> Dict[str, NDArray]:
        dir_path = self.dataset_dir
        csv_path = os.path.join(dir_path, "imu_data.csv")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"IMU CSV file not found: {csv_path}")
        data = pd.read_csv(csv_path)
        return {
            "frame_ids": data["frame"].to_numpy(np.int32),
            "times": data["time"].to_numpy(np.float32),
            "drone_pos_xyz": data[["pos_x", "pos_y", "pos_z"]].to_numpy(np.float32),
            "drone_quat_xyzw": data[["quat_x", "quat_y", "quat_z", "quat_w"]].to_numpy(np.float32),
            "drone_angvel": data[["angvel_x", "angvel_y", "angvel_z"]].to_numpy(np.float32),
            "drone_accel": data[["accel_x", "accel_y", "accel_z"]].to_numpy(np.float32),
        }

    def load_img_paths(self) -> List[str]:
        img_dir = os.path.join(self.dataset_dir, "imgs")
        return [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".jpg")]


def load_config(file_path: Path) -> SceneConfig:
    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return SceneConfig.from_dict(file_path, data)


def save_config(config: SceneConfig, file_path: Path) -> None:
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=4, ensure_ascii=False)


def compute_gate_in_camera(
    gate_pos_xyz: NDArray[np.float32],
    gate_quat_xyzw: NDArray[np.float32],
    drone_pos_xyz: NDArray[np.float32],
    drone_quat_xyzw: NDArray[np.float32],
    R_cam_to_drone: NDArray[np.float32],
    T_cam_to_drone: NDArray[np.float32],
) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
    r_gate_world = Rotation.from_quat(gate_quat_xyzw)
    r_drone_world = Rotation.from_quat(drone_quat_xyzw)
    r_world_to_drone = r_drone_world.inv()
    r_cam_to_body = Rotation.from_matrix(R_cam_to_drone)
    r_body_to_cam = r_cam_to_body.inv()
    r_WD = r_drone_world.inv()
    r_gate_in_body = r_gate_world * r_WD
    r_gate_in_camera = r_gate_in_body * r_cam_to_body
    rel_pos_world = gate_pos_xyz - drone_pos_xyz
    pos_body = r_world_to_drone.apply(rel_pos_world)
    pos_body_cam_offset = pos_body - T_cam_to_drone
    pos_cam = r_body_to_cam.apply(pos_body_cam_offset)
    euler_cam = r_gate_in_camera.as_euler("xyz", degrees=True)
    pos_cam = pos_cam.astype(np.float32)
    euler_cam = euler_cam.astype(np.float32)
    return pos_cam, euler_cam


# ========== 新增：从附件1复制的 order_points 和 3D 参考点 ==========
def order_points(pts):
    """将4个点排序为 [右下, 右上, 左上, 左下]（与附件1一致）"""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmax(s)]   # 右下 (sum最大)
    rect[1] = pts[np.argmax(diff)] # 右上 (diff最大)
    rect[2] = pts[np.argmin(s)]   # 左上 (sum最小)
    rect[3] = pts[np.argmin(diff)] # 左下 (diff最小)
    return rect


# 附件1中使用的3D参考点（门框在XY平面，法线沿Z，半宽高1.35m）
RECTANGLE_3D_POINTS = np.array([
    [1.35,  1.35, 0],
    [-1.35, 1.35, 0],
    [-1.35, -1.35, 0],
    [1.35, -1.35, 0]
], dtype=np.float32)
# ================================================================


def main(dataset_dir: Path, yolo_model_path: str = "best.pt"):
    meta = load_config(dataset_dir / "metadata.json")
    start_idx = 0
    end_idx = 48
    img_paths = meta.load_img_paths()[start_idx:end_idx]
    imu_data = meta.load_imu_data()
    frame_ids: NDArray[np.int32] = imu_data["frame_ids"][start_idx:end_idx]
    times: NDArray[np.float32] = imu_data["times"][start_idx:end_idx]
    drone_pos_xyz: NDArray[np.float32] = imu_data["drone_pos_xyz"][start_idx:end_idx]
    drone_quat_xyzw: NDArray[np.float32] = imu_data["drone_quat_xyzw"][start_idx:end_idx]
    drone_angvel: NDArray[np.float32] = imu_data["drone_angvel"][start_idx:end_idx]
    drone_accel: NDArray[np.float32] = imu_data["drone_accel"][start_idx:end_idx]

    output_dir = Path(__file__).parent.parent / "output" / dataset_dir.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    output_img_dir = output_dir / "img"
    output_img_dir.mkdir(parents=True, exist_ok=True)

    # ----- 构建项目B风格的3D点（按BR,TR,TL,BL） -----
    w = meta.gate.dimensions[1]
    h = meta.gate.dimensions[2]
    gate_points_3d = np.array([
        [0, -w/2, -h/2],
        [0, -w/2,  h/2],
        [0,  w/2,  h/2],
        [0,  w/2, -h/2],
    ], dtype=np.float64)

    extrinsic_matrix = np.array(
        [
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    R_cam_to_drone = extrinsic_matrix[:3, :3].T
    T_cam_to_drone = extrinsic_matrix[:3, 3].T

    gate_pos_xyz = np.asarray(meta.gate.position, dtype=np.float32)
    gate_euler_xyz_rad = np.asarray(meta.gate.euler_xyz_rad, dtype=np.float32)
    gate_quat_xyzw = Rotation.from_euler("xyz", gate_euler_xyz_rad).as_quat()

    # ---------- 替换：不再使用 pipeline，直接用 YOLO ----------
    print(f"Loading YOLO model from {yolo_model_path} ...")
    yolo = YOLO(yolo_model_path)

    # 使用数据集中的相机内参（畸变系数设为零，与项目B一致）
    camera_matrix = meta.camera.camera_matrix.astype(np.float64)
    dist_coeffs = np.zeros(5, dtype=np.float64)

    n_frames = len(img_paths)
    gate_pos_est_list = []   # 存储估计位置
    gate_pos_true_list = []  # 存储真实位置（相机系）
    gate_roll_est, gate_pitch_est, gate_yaw_est = [], [], []
    gate_roll_true, gate_pitch_true, gate_yaw_true = [], [], []

    print(f"门框在地面系真实位姿\nposition: {gate_pos_xyz}\n"
          f"orientation: {Rotation.from_quat(gate_quat_xyzw).as_euler('xyz', degrees=True)}")

    for i, img_path in enumerate(img_paths):
        frame_id = frame_ids[i]
        print(f"===== Processing frame {frame_id} =====")
        raw_img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        raw_img_rgb = cv2.cvtColor(raw_img_bgr, cv2.COLOR_BGR2RGB)

        # 真实门框在相机系位姿
        gate_pos_cam, gate_euler_cam = compute_gate_in_camera(
            gate_pos_xyz, gate_quat_xyzw,
            drone_pos_xyz[i], drone_quat_xyzw[i],
            R_cam_to_drone, T_cam_to_drone,
        )
        gate_pos_true_list.append(gate_pos_cam)
        gate_roll_true.append(gate_euler_cam[0])
        gate_pitch_true.append(gate_euler_cam[1])
        gate_yaw_true.append(gate_euler_cam[2])
        print(f"门框在相机系真实位姿\nposition: {gate_pos_cam}\norientation: {gate_euler_cam}")

        # ---------- YOLO 预测 ----------
        results = yolo(
            source=raw_img_rgb,
            conf=0.5,
            iou=0.7,
            max_det=10,
            half=True,
            rect=False,
            verbose=False,
        )

        # ---------- 提取 mask 轮廓并进行四边形检测（附件1风格） ----------
        best_pose = None
        min_distance = float('inf')

        for result in results:
            if hasattr(result, "masks") and result.masks is not None:
                for mask_tensor in result.masks:
                    mask_points = mask_tensor.xy[0].tolist()
                    mask_array = np.array([[int(p[0]), int(p[1])] for p in mask_points], dtype=np.int32)
                    epsilon = 0.05 * cv2.arcLength(mask_array, True)
                    approx = cv2.approxPolyDP(mask_array, epsilon, True)
                    if len(approx) == 4:
                        pts = np.array([point[0] for point in approx], dtype=np.float32)
                        ordered_pts = order_points(pts)   # 顺序：[右下,右上,左上,左下]
                        success, rvec, tvec = cv2.solvePnP(
                            gate_points_3d,               # 使用项目B的3D点
                            ordered_pts.astype(np.float64),
                            camera_matrix,
                            dist_coeffs,
                            flags=cv2.SOLVEPNP_SQPNP
                        )
                        if success:
                            tvec = tvec.flatten()
                            dist = np.linalg.norm(tvec)
                            if dist < min_distance:
                                min_distance = dist
                                R, _ = cv2.Rodrigues(rvec)
                                r = Rotation.from_matrix(R)
                                quat = r.as_quat()
                                best_pose = {
                                    'position': tvec,
                                    'orientation': quat,
                                    'rvec': rvec.flatten(),
                                    'tvec': tvec
                                }

        # ---------- 处理估计结果 ----------
        if best_pose is not None:
            pos_est = best_pose['position']
            quat_est = best_pose['orientation']
            euler_est = Rotation.from_quat(quat_est).as_euler('xyz', degrees=True)  # 与项目A打印一致
            gate_pos_est_list.append(pos_est)
            gate_roll_est.append(euler_est[0])
            gate_pitch_est.append(euler_est[1])
            gate_yaw_est.append(euler_est[2])
            pos_err = np.linalg.norm(pos_est - gate_pos_cam)
            print(f"估计位姿（相机系）\nposition: {pos_est}\norientation (zyx deg): {euler_est}")
            print(f"位置误差: {pos_err:.4f} m ({pos_err / np.linalg.norm(gate_pos_cam) * 100:.2f}%)")
        else:
            print("未检测到四边形门框")
            # 填充NaN以便绘图
            gate_pos_est_list.append([np.nan, np.nan, np.nan])
            gate_roll_est.append(np.nan)
            gate_pitch_est.append(np.nan)
            gate_yaw_est.append(np.nan)

    # ---------- 绘图（同原版） ----------
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    ax = axes.flatten()

    ax[0].plot(gate_roll_true, label='True')
    ax[0].plot(gate_roll_est, label='Est')
    ax[0].set_title("Roll")
    ax[0].set_xlabel("Frame ID")
    ax[0].set_ylabel("Roll Angle (deg)")
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(gate_pitch_true, label='True')
    ax[1].plot(gate_pitch_est, label='Est')
    ax[1].set_title("Pitch")
    ax[1].set_xlabel("Frame ID")
    ax[1].set_ylabel("Pitch Angle (deg)")
    ax[1].legend()
    ax[1].grid()

    ax[2].plot(gate_yaw_true, label='True')
    ax[2].plot(gate_yaw_est, label='Est')
    ax[2].set_title("Yaw")
    ax[2].set_xlabel("Frame ID")
    ax[2].set_ylabel("Yaw Angle (deg)")
    ax[2].legend()
    ax[2].grid()

    gate_pos_true_arr = np.array(gate_pos_true_list)
    gate_pos_est_arr = np.array(gate_pos_est_list)
    for idx, (label, data_true, data_est) in enumerate(
        zip(['x', 'y', 'z'],
            [gate_pos_true_arr[:, 0], gate_pos_true_arr[:, 1], gate_pos_true_arr[:, 2]],
            [gate_pos_est_arr[:, 0], gate_pos_est_arr[:, 1], gate_pos_est_arr[:, 2]])
    ):
        ax[3+idx].plot(data_true, label='True')
        ax[3+idx].plot(data_est, label='Est')
        ax[3+idx].set_title(f"Position ({label})")
        ax[3+idx].set_xlabel("Frame ID")
        ax[3+idx].set_ylabel("Position (m)")
        ax[3+idx].legend()
        ax[3+idx].grid()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 可以指定模型路径，例如 "best-seg.engine" 或 "best.pt"
    main(Path(r"D:\Documents\BIT\2025-2026_2\a2rl\monorace_perception\assets\anim_test-1"), 
         yolo_model_path=r"models\best.pt")