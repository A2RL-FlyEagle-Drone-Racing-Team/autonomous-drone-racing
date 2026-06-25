# region 读取数据集
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

from src.pipeline.vision_pipeline_yolo import VisionRacingYOLOPipeline, YOLOPipelineConfig
from src.vision.pose_estimator import PoseEstimator


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
        """
        从 JSON 所在目录读取 imu_data.csv，并将数据存储为实例属性。
        返回包含所有数组的字典，方便直接使用。
        """
        dir_path = self.dataset_dir
        csv_path = os.path.join(dir_path, "imu_data.csv")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"IMU CSV file not found: {csv_path}")

        # 读取 CSV（跳过表头）
        # 使用 names=True 自动从第一行读取列名，返回结构化数组
        data = pd.read_csv(csv_path)

        # 返回字典以便快速使用
        return {
            "frame_ids": data["frame"].to_numpy(np.int32),
            "times": data["time"].to_numpy(np.float32),
            "drone_pos_xyz": data[["pos_x", "pos_y", "pos_z"]].to_numpy(np.float32),
            "drone_quat_xyzw": data[["quat_x", "quat_y", "quat_z", "quat_w"]].to_numpy(np.float32),
            "drone_angvel": data[["angvel_x", "angvel_y", "angvel_z"]].to_numpy(np.float32),
            "drone_accel": data[["accel_x", "accel_y", "accel_z"]].to_numpy(np.float32),
        }

    def load_img_paths(self) -> List[str]:
        """
        读取图像文件路径
        """
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
    """
    计算门框在相机系（右下前）中的位置和姿态欧拉角（度）
    """
    # 创建旋转对象
    r_gate_world = Rotation.from_quat(gate_quat_xyzw)  # 世界→门
    r_drone_world = Rotation.from_quat(drone_quat_xyzw)  # 世界→机体
    r_world_to_drone = r_drone_world.inv()

    # 相机→机体 及其逆
    r_cam_to_body = Rotation.from_matrix(R_cam_to_drone)  # 相机→机体
    r_body_to_cam = r_cam_to_body.inv()  # 机体→相机

    # 1. 计算门在相机系中的姿态（从相机到门的旋转）
    #    r_WD = 机体→世界 = (世界→机体)^(-1)
    r_WD = r_drone_world.inv()
    #    门在机体中的姿态：r_GD = (世界→门) * (机体→世界)
    r_gate_in_body = r_gate_world * r_WD
    #    门在相机中的姿态：r_GC = (机体→门) * (相机→机体)
    r_gate_in_camera = r_gate_in_body * r_cam_to_body

    # 2. 计算门在相机系中的位置
    rel_pos_world = gate_pos_xyz - drone_pos_xyz  # 门相对于无人机的世界坐标
    pos_body = r_world_to_drone.apply(rel_pos_world)  # 转到机体坐标系
    pos_body_cam_offset = pos_body - T_cam_to_drone  # 减去相机在机体中的偏移
    pos_cam = r_body_to_cam.apply(pos_body_cam_offset)  # 转到相机坐标系

    # 3. 提取欧拉角 (roll, pitch, yaw)，顺序 'xyz'
    euler_cam = r_gate_in_camera.as_euler("xyz", degrees=True)

    # 转换为 float32
    pos_cam = pos_cam.astype(np.float32)
    euler_cam = euler_cam.astype(np.float32)

    return pos_cam, euler_cam


# endregion


# region 主函数
def main(dataset_dir: Path, run_pipeline: bool = True):
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

    pipeline = None
    if run_pipeline:
        pose_estimator = PoseEstimator(
            gate_width=meta.gate.dimensions[1],
            gate_height=meta.gate.dimensions[2],
            camera_matrix=meta.camera.camera_matrix,
            dist_coeffs=np.zeros(5, dtype=np.float32),
            image_size=(meta.camera.resolution_x, meta.camera.resolution_y),
            camera_fov=None,
            gate_points_3d=meta.gate.gate_points_3d,
        )
        print(pose_estimator.gate_points_3d)
        pipeline_config = YOLOPipelineConfig(
            vision_freq=meta.fps,
            gate_width=meta.gate.dimensions[1] / 2,
            gate_height=meta.gate.dimensions[2] / 2,
            image_width=meta.camera.resolution_x,
            image_height=meta.camera.resolution_y,
            camera_fov=None,
            intrinsic_matrix=meta.camera.camera_matrix,
            extrinsic_matrix=np.array(
                [
                    [0, -1, 0, 0],
                    [0, 0, -1, 0],
                    [1, 0, 0, 0],
                    [0, 0, 0, 1],
                ],
                dtype=np.float32,
            ),
            dist_coeffs=np.zeros(5, dtype=np.float32),
            yolo_model_name="best.pt",
        )
        pipeline = VisionRacingYOLOPipeline(
            pipeline_config,
            pose_estimator,
            known_gates={0: (gate_pos_xyz, gate_euler_xyz_rad)},
        )
        pipeline.reset(
            initial_position=drone_pos_xyz[0],
            initial_velocity=np.zeros(3, dtype=np.float32),
            initial_orientation=drone_quat_xyzw[0],
        )

    n_frames = len(img_paths)
    cam_positions_output = []
    gate_roll = []
    gate_pitch = []
    gate_yaw = []
    gate_roll_est = []
    gate_pitch_est = []
    gate_yaw_est = []
    gate_pos = []
    gate_pos_est = []
    print(
        f"门框在地面系真实位姿\nposition: {gate_pos_xyz}\n"
        f"orientation: {Rotation.from_quat(gate_quat_xyzw).as_euler('xyz', degrees=True)}"
    )

    for i, img_path in enumerate(img_paths):
        frame_id = frame_ids[i]
        dt = 1 / meta.fps
        print(f"===== Processing frame {frame_id} =====")
        raw_img_rgb = np.asarray(cv2.imread(img_path, cv2.IMREAD_COLOR_BGR))
        print(
            f"机体在地面系真实位姿\nposition: {drone_pos_xyz[i]}\n"
            f"orientation: {Rotation.from_quat(drone_quat_xyzw[i]).as_euler('xyz', degrees=True)}"
        )
        gate_pos_cam, gate_euler_cam = compute_gate_in_camera(
            gate_pos_xyz,
            gate_quat_xyzw,
            drone_pos_xyz[i],
            drone_quat_xyzw[i],
            R_cam_to_drone,
            T_cam_to_drone,
        )
        gate_roll.append(gate_euler_cam[0])
        gate_pitch.append(gate_euler_cam[1])
        gate_yaw.append(gate_euler_cam[2])
        gate_pos.append(gate_pos_cam)

        print(f"门框在相机系位姿\nposition: {gate_pos_cam}\norientation: {gate_euler_cam}")

        if run_pipeline and pipeline is not None:
            gate_pose = pipeline.process_image(
                raw_img_rgb,
                current_time=times[i],
                save_mask_path=str(output_img_dir / f"{frame_id}.png"),
            )
            if gate_pose is not None:
                pos_est_err = np.linalg.norm(gate_pose.position - gate_pos_cam)
                gate_pose_orientation = Rotation.from_quat(gate_pose.orientation).as_euler("zyx", degrees=True)
                gate_roll_est.append(gate_pose_orientation[0])
                gate_pitch_est.append(gate_pose_orientation[1])
                gate_yaw_est.append(gate_pose_orientation[2])
                gate_pos_est.append(gate_pose.position)
                print(
                    f"门框在相机系位姿估计\nposition: {gate_pose.position}\n"
                    f"orientation: {gate_pose_orientation}\n"
                    f"重投影误差: {gate_pose.reprojection_error}\n"
                    f"位姿估计误差: {pos_est_err} ({pos_est_err / np.linalg.norm(gate_pos_cam) * 100:.2f} %)"
                )  # TODO: 欧拉角顺序？（rvec -> euler）
            if gate_pose is not None:
                pipeline.update_state_with_vision(gate_pose, 0)
            # state = pipeline.get_state()
            # state = pipeline.predict_state(dt, drone_accel[i], drone_angvel[i])
            # cam_positions_output.append(state)

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    ax = axes.flatten()
    ax[0].plot(gate_roll)
    ax[0].plot(gate_roll_est)
    ax[0].set_title("Roll")
    ax[0].set_xlabel("Frame ID")
    ax[0].set_ylabel("Roll Angle (deg)")
    ax[0].legend(["True", "Est"])
    ax[0].grid()

    ax[1].plot(gate_pitch)
    ax[1].plot(gate_pitch_est)
    ax[1].set_title("Pitch")
    ax[1].set_xlabel("Frame ID")
    ax[1].set_ylabel("Pitch Angle (deg)")
    ax[1].legend(["True", "Est"])
    ax[1].grid()

    ax[2].plot(gate_yaw)
    ax[2].plot(gate_yaw_est)
    ax[2].set_title("Yaw")
    ax[2].set_xlabel("Frame ID")
    ax[2].set_ylabel("Yaw Angle (deg)")
    ax[2].legend(["True", "Est"])
    ax[2].grid()

    gate_pos = np.asarray(gate_pos)
    gate_pos_est = np.asarray(gate_pos_est)
    ax[3].plot(gate_pos[:, 0])
    ax[3].plot(gate_pos_est[:, 0])
    ax[3].set_title("Position (x)")
    ax[3].set_xlabel("Frame ID")
    ax[3].set_ylabel("Position (m)")
    ax[3].legend(["True", "Est"])
    ax[3].grid()

    ax[4].plot(gate_pos[:, 1])
    ax[4].plot(gate_pos_est[:, 1])
    ax[4].set_title("Position (y)")
    ax[4].set_xlabel("Frame ID")
    ax[4].set_ylabel("Position (m)")
    ax[4].legend(["True", "Est"])
    ax[4].grid()

    ax[5].plot(gate_pos[:, 2])
    ax[5].plot(gate_pos_est[:, 2])
    ax[5].set_title("Position (z)")
    ax[5].set_xlabel("Frame ID")
    ax[5].set_ylabel("Position (m)")
    ax[5].legend(["True", "Est"])
    ax[5].grid()

    plt.tight_layout()
    plt.show()


# endregion

if __name__ == "__main__":
    main(Path(r"D:\Documents\BIT\2025-2026_2\a2rl\monorace_perception\assets\anim_test-1"))
