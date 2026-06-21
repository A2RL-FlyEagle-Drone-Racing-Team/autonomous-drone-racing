from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
from loguru import logger
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.spatial.transform import Rotation
import yaml

# from scipy.spatial.transform import Rotation
# from loguru import logger

from src.pipeline.vision_pipeline_yolo import VisionRacingYOLOPipeline, YOLOPipelineConfig
from src.vision.pose_estimator import PoseEstimator


# region 读取数据集
@dataclass(frozen=True)
class DatasetMeta:
    """
    Params:
        cam_resolution_x (int): 相机分辨率宽度（px）
        cam_resolution_y (int): 相机分辨率高度（px）
        fps (int): 视频帧率（fps）
        focal_length (float): 相机焦距（mm）
        sensor_size (float): 相机传感器尺寸（mm）
        gate_pos (NDArray): 门框位置（m）
        gate_euler (NDArray): 门框欧拉角（deg）
        gate_out_w (float): 门框外部宽度（m）
        gate_out_h (float): 门框外部高度（m）
        gate_in_w (float): 门框内部宽度（m）
        gate_in_h (float): 门框内部高度（m）
    """

    cam_resolution_x: int
    cam_resolution_y: int
    fps: int
    focal_length: float
    sensor_size: float
    gate_pos: NDArray[np.float64]
    gate_euler: NDArray[np.float64]
    gate_out_w: float
    gate_out_h: float
    gate_in_w: float
    gate_in_h: float

    @property
    def intrinsic_matrix(self):
        """
        计算相机内参矩阵
        > 相机系：右上后
        """
        return np.array(
            [
                [self.focal_length * self.cam_resolution_x / self.sensor_size, 0.0, self.cam_resolution_x / 2.0],
                [0.0, self.focal_length * self.cam_resolution_y / self.sensor_size, self.cam_resolution_y / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )


def load_meta(dataset_dir: Path):
    """
    读取数据集元数据
    """
    path = dataset_dir / "meta.yaml"
    with open(path, "r", encoding="utf-8") as f:
        meta = yaml.safe_load(f)
    return DatasetMeta(**meta)


def load_cam_data(dataset_dir: Path) -> Tuple[NDArray[np.int_], NDArray[np.float64], NDArray[np.float64]]:
    """
    读取相机数据

    Returns:
        Tuple[NDArray[int], NDArray[float64], NDArray[float64]]: (frame_id, cam_pos_xyz, cam_quat_xyzw)
    """
    path = dataset_dir / "camera_data.csv"
    df = pd.read_csv(path)

    # 提取帧ID（整数）
    frame_id = df["frame"].to_numpy(dtype=np.int_)

    # 提取位置坐标（N x 3）
    cam_pos_xyz = df[["loc_x", "loc_y", "loc_z"]].to_numpy(dtype=np.float64)

    # 提取四元数并转换为 (x, y, z, w) 顺序
    # CSV中顺序为 w, x, y, z，目标顺序为 x, y, z, w
    cam_quat_xyzw = df[["quat_x", "quat_y", "quat_z", "quat_w"]].to_numpy(dtype=np.float64)

    return frame_id, cam_pos_xyz, cam_quat_xyzw


def load_img_paths(dataset_dir: Path) -> List[str]:
    """
    读取图像路径
    """
    path = dataset_dir / "img"
    return [str(img) for img in path.glob("*.jpg")]


# endregion


def compute_gate_in_camera(
    gate_pos_xyz: NDArray[np.float64],
    gate_euler_xyz_deg: NDArray[np.float64],
    cam_pos_xyz: NDArray[np.float64],
    cam_quat_xyzw: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    计算门框在相机坐标系（右下前）中的位置
    """
    R_cam: NDArray[np.float64] = Rotation.from_quat(cam_quat_xyzw).as_matrix()
    R_world_to_cam = R_cam.T
    gate_pos_cam_xyz = R_world_to_cam @ (gate_pos_xyz - cam_pos_xyz)
    R_gate: NDArray[np.float64] = Rotation.from_euler("xyz", gate_euler_xyz_deg, degrees=True).as_matrix()
    R_gate_cam = R_world_to_cam @ R_gate
    gate_euler_cam_xyz_deg = Rotation.from_matrix(R_gate_cam).as_euler("xyz", degrees=True)
    return gate_pos_cam_xyz, gate_euler_cam_xyz_deg


# region 主函数
def main(dataset_dir: Path):
    meta = load_meta(dataset_dir)
    max_count = 1
    img_paths = load_img_paths(dataset_dir)[:max_count]
    frame_ids, cam_positions_xyz, cam_quaternions_xyzw = load_cam_data(dataset_dir)
    frame_ids = frame_ids[:max_count]
    cam_positions_xyz = cam_positions_xyz[:max_count]
    cam_quaternions_xyzw = cam_quaternions_xyzw[:max_count]

    output_dir = Path(__file__).parent.parent / "output" / dataset_dir.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    output_img_dir = output_dir / "img"
    output_img_dir.mkdir(parents=True, exist_ok=True)

    pose_estimator = PoseEstimator(
        gate_width=meta.gate_out_w,
        gate_height=meta.gate_out_h,
        camera_matrix=meta.intrinsic_matrix,
        dist_coeffs=np.zeros(5, dtype=np.float64),
        image_size=(meta.cam_resolution_x, meta.cam_resolution_y),
        camera_fov=None,
        gate_points_3d=np.array(
            [
                [0, -meta.gate_in_w / 2, meta.gate_in_h / 2],
                [0, meta.gate_in_w / 2, meta.gate_in_h / 2],
                [0, meta.gate_in_w / 2, -meta.gate_in_h / 2],
                [0, -meta.gate_in_w / 2, -meta.gate_in_h / 2],
            ],
            dtype=np.float64,
        ),
    )

    pipeline_config = YOLOPipelineConfig(
        vision_freq=meta.fps,
        gate_width=meta.gate_out_w,
        gate_height=meta.gate_out_h,
        image_width=meta.cam_resolution_x,
        image_height=meta.cam_resolution_y,
        camera_fov=None,
        intrinsic_matrix=meta.intrinsic_matrix,
        extrinsic_matrix=np.diag(np.array([1, -1, -1, 1], dtype=np.float64)),
        dist_coeffs=np.zeros(5, dtype=np.float64),
        yolo_model_name="best.pt",
    )
    pipeline = VisionRacingYOLOPipeline(
        pipeline_config,
        pose_estimator=pose_estimator,
        known_gates={0: (meta.gate_pos, meta.gate_euler)},
    )
    pipeline.reset(
        initial_position=cam_positions_xyz[0],
        initial_orientation=cam_quaternions_xyzw[0],
        initial_velocity=np.zeros(3, dtype=np.float64),
    )

    n_frames = len(img_paths)
    tspan = np.linspace(0.0, n_frames / meta.fps, n_frames)
    cam_positions_output = []
    for i, img_path in enumerate(img_paths):
        logger.debug(f"====== Processing frame {i+1} ======")
        frame_id = frame_ids[i]
        raw_img_rgb = np.asarray(cv2.imread(img_path, cv2.IMREAD_COLOR_BGR), dtype=np.uint8)
        logger.debug(f"[门框在地面系真实位姿]\nposition: {meta.gate_pos}, orientation: {meta.gate_euler}")
        logger.debug(
            f"[相机在地面系真实位姿]\nposition: {cam_positions_xyz[i]}, "
            f"orientation: {Rotation.from_quat(cam_quaternions_xyzw[i]).as_euler('xyz', True)}"
        )

        gate_pos_camera, gate_rot_camera = compute_gate_in_camera(
            meta.gate_pos, meta.gate_euler, cam_positions_xyz[i], cam_quaternions_xyzw[i]
        )
        logger.debug(f"[门框在相机系（右下前）真实位姿]\nposition: {gate_pos_camera}, orientation: {gate_rot_camera}")

        gate_pose = pipeline.process_image(
            raw_img_rgb,
            tspan[i],
            save_mask_path=str(output_img_dir / f"{frame_id:04d}.jpg"),
        )
        if gate_pose is not None:
            pipeline.update_state_with_vision(gate_pose, 0)
        state = pipeline.get_state()
        cam_positions_output.append(state.position)

    cam_positions_output_arr = np.array(cam_positions_output, dtype=np.float64)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    ax: Axes
    ylabels = ["x", "y", "z"]
    for i, ax in enumerate(axes.flatten()):
        ax.plot(tspan, cam_positions_xyz[:, i])
        ax.plot(tspan, cam_positions_output_arr[:, i])
        ax.set_xlabel(f"time (s)")
        ax.set_ylabel(ylabels[i] + " (m)")
        ax.legend(["True Cam Pos", "Updated Cam Pos"])
        ax.grid()

    fig.tight_layout()
    fig.savefig(output_dir / "state_position.png")


# endregion

if __name__ == "__main__":
    main(Path(r"D:\Documents\BIT\2025-2026_2\a2rl\monorace_perception\assets\binary_test-2"))
