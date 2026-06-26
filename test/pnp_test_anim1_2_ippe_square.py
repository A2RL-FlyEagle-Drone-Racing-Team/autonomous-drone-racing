"""
IPPE_SQUARE + 候选验证 + 保留 R_adj 方案
因 R_adj 不匹配 IPPE_SQUARE 的轴约定，导致位置和姿态完全错误
"""
import os
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass
from scipy.spatial.transform import Rotation
from ultralytics import YOLO


# ---------- 核心类（从附件中抽离，未改变逻辑） ----------
@dataclass
class GateDetection:
    corners: np.ndarray
    confidence: float
    is_complete: bool
    visible_corners: int
    contour: Optional[np.ndarray] = None
    center: Optional[np.ndarray] = None


class QuAdGate:
    def __init__(self, min_contour_area=100, epsilon_factor=0.02, max_corner_distance=10.0, confidence_threshold=0.5):
        self.min_contour_area = min_contour_area
        self.epsilon_factor = epsilon_factor
        self.max_corner_distance = max_corner_distance
        self.confidence_threshold = confidence_threshold

    def detect(self, mask: np.ndarray) -> Optional[GateDetection]:
        if mask.dtype in (np.float32, np.float64):
            mask = (mask * 255).astype(np.uint8)
        if mask.max() > 1:
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_contour_area:
                continue
            det = self._process_contour(cnt, mask.shape)
            if det is not None:
                detections.append(det)
        if not detections:
            return None
        return max(detections, key=lambda d: d.confidence)

    def _process_contour(self, contour, image_shape):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return None
        epsilon = self.epsilon_factor * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        num_vertices = len(approx)
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            return None
        center = np.array([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]])
        if num_vertices == 4:
            corners = approx.reshape(4, 2).astype(np.float32)
            corners = self._order_corners(corners)
            conf = self._compute_confidence(corners, area, perimeter)
            return GateDetection(corners, conf, True, 4, contour, center)
        elif num_vertices == 3:
            corners = approx.reshape(3, 2).astype(np.float32)
            corners = self._complete_from_triangle(corners, center)
            conf = self._compute_confidence(corners, area, perimeter) * 0.7
            return GateDetection(corners, conf, False, 3, contour, center)
        else:
            rect = cv2.minAreaRect(contour)
            corners = cv2.boxPoints(rect).astype(np.float32)
            corners = self._order_corners(corners)
            conf = self._compute_confidence(corners, area, perimeter) * (0.8 if num_vertices > 4 else 0.5)
            return GateDetection(corners, conf, num_vertices == 4, min(num_vertices, 4), contour, center)

    def _order_corners(self, corners):
        sum_coords = corners.sum(axis=1)
        tl_idx = np.argmin(sum_coords)
        br_idx = np.argmax(sum_coords)
        diff_coords = np.diff(corners, axis=1).flatten()
        tr_idx = np.argmin(diff_coords)
        bl_idx = np.argmax(diff_coords)
        indices = [tl_idx, tr_idx, br_idx, bl_idx]
        if len(set(indices)) != 4:
            center = corners.mean(axis=0)
            angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
            order = np.argsort(angles)
            start = np.argmin(corners[order].sum(axis=1))
            order = np.roll(order, -start)
            return corners[order]
        return corners[[tl_idx, tr_idx, br_idx, bl_idx]]

    def _complete_from_triangle(self, corners, center):
        best_fourth = None
        best_score = float("inf")
        for i in range(3):
            p1, p2, p3 = corners[i], corners[(i + 1) % 3], corners[(i + 2) % 3]
            p4 = p1 + p3 - p2
            all_corners = np.vstack([corners, p4.reshape(1, 2)])
            score = self._rectangularity_score(all_corners)
            if score < best_score:
                best_score = score
                best_fourth = p4
        if best_fourth is None:
            farthest = np.argmax(np.linalg.norm(corners - center, axis=1))
            best_fourth = 2 * center - corners[farthest]
        all_corners = np.vstack([corners, best_fourth.reshape(1, 2)])
        return self._order_corners(all_corners)

    def _rectangularity_score(self, corners):
        ordered = self._order_corners(corners)
        sides = np.array(
            [ordered[1] - ordered[0], ordered[2] - ordered[1], ordered[3] - ordered[2], ordered[0] - ordered[3]]
        )
        length_diff = abs(np.linalg.norm(sides[0]) - np.linalg.norm(sides[2]))
        length_diff += abs(np.linalg.norm(sides[1]) - np.linalg.norm(sides[3]))
        angle_dev = 0.0
        for i in range(4):
            v1 = sides[i]
            v2 = sides[(i + 1) % 4]
            cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            angle_dev += abs(cos_a)
        return length_diff + angle_dev

    def _compute_confidence(self, corners, area, perimeter):
        rect_score = self._rectangularity_score(corners)
        rect_conf = max(0, 1 - rect_score / 10)
        min_area, max_area = 500, 80000
        if area < min_area:
            area_conf = area / min_area
        elif area > max_area:
            area_conf = max_area / area
        else:
            area_conf = 1.0
        ordered = self._order_corners(corners)
        width = np.linalg.norm(ordered[1] - ordered[0])
        height = np.linalg.norm(ordered[3] - ordered[0])
        aspect = width / (height + 1e-6)
        aspect_conf = 1.0 if 0.5 <= aspect <= 2.0 else 0.5
        return float(np.clip(rect_conf * 0.4 + area_conf * 0.3 + aspect_conf * 0.3, 0, 1))


@dataclass
class GatePose:
    position: np.ndarray
    orientation: np.ndarray  # xyzw
    rotation_matrix: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    confidence: float
    reprojection_error: float
    distance: float


class PoseEstimator:
    def __init__(self, gate_width=2.7, gate_height=2.7, camera_matrix=None, dist_coeffs=None, image_size=(384, 384)):
        self.gate_width = gate_width
        self.gate_height = gate_height
        self.image_size = image_size
        if camera_matrix is None:
            raise ValueError("camera_matrix must be provided")
        self.camera_matrix = camera_matrix.astype(np.float64)
        self.dist_coeffs = np.zeros(5, dtype=np.float64) if dist_coeffs is None else dist_coeffs.astype(np.float64)
        # 3D points in gate frame: TL, TR, BR, BL (z=0 plane)
        # self.gate_points_3d = np.array(
        #     [
        #         [-gate_width / 2, -gate_height / 2, 0],
        #         [gate_width / 2, -gate_height / 2, 0],
        #         [gate_width / 2, gate_height / 2, 0],
        #         [-gate_width / 2, gate_height / 2, 0],
        #     ],
        #     dtype=np.float64,
        # )
        self.gate_points_3d = np.array(
            [
                [-gate_width / 2, -gate_height / 2, 0],  # TL
                [gate_width / 2, -gate_height / 2, 0],  # TR
                [gate_width / 2, gate_height / 2, 0],  # BR
                [-gate_width / 2, gate_height / 2, 0],  # BL
            ],
            dtype=np.float64,
        )

        # 定义固定的轴校正矩阵（IPPE 专用）
        self.R_adj = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float64)

    def estimate_pose(self, detection: GateDetection) -> Optional[GatePose]:
        if detection is None or detection.corners is None:
            return None
        image_points = detection.corners.astype(np.float64)
        if len(image_points) != 4:
            return None
        try:
            # 使用 IPPE_SQUARE 求解（返回一个解）
            success, rvec_ippe, tvec = cv2.solvePnP(
                self.gate_points_3d, image_points, self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            if not success:
                return None

            # ----- 应用 R_adj 校正旋转 -----
            R_ippe, _ = cv2.Rodrigues(rvec_ippe)
            R_corrected = R_ippe @ self.R_adj  # 校正后的旋转矩阵（标准坐标系）
            rvec_corrected, _ = cv2.Rodrigues(R_corrected)
            tvec = tvec.flatten()

            # ----- 生成 4 个旋转对称候选 -----
            # 绕门平面法线（z轴）旋转 0°, 90°, 180°, 270°
            candidates = []
            # 绕门平面法线（即校正后的 Z 轴）旋转 0°, 90°, 180°, 270°
            for angle_deg in [0, 90, 180, 270]:
                angle_rad = np.deg2rad(angle_deg)
                # 绕 Z 轴旋转矩阵
                Rz = cv2.Rodrigues(np.array([0, 0, angle_rad]))[0]
                R_candidate = R_corrected @ Rz
                rvec_candidate, _ = cv2.Rodrigues(R_candidate)
                # 计算重投影误差
                proj, _ = cv2.projectPoints(
                    self.gate_points_3d, rvec_candidate, tvec, self.camera_matrix, self.dist_coeffs
                )
                proj = proj.reshape(-1, 2)
                reproj_err = np.mean(np.linalg.norm(proj - image_points, axis=1))
                candidates.append((rvec_candidate.flatten(), tvec, reproj_err))

            # 选择重投影误差最小的候选
            best_rvec, best_tvec, best_err = min(candidates, key=lambda x: x[2])

            # ----- 后续处理 -----
            R, _ = cv2.Rodrigues(best_rvec)
            q = self._rot_to_quat(R)
            dist = np.linalg.norm(best_tvec)
            conf = detection.confidence * max(0, 1 - best_err / 10)
            return GatePose(best_tvec, q, R, best_rvec, best_tvec, conf, best_err, float(dist))
        except cv2.error:
            return None

    def _rot_to_quat(self, R):
        trace = np.trace(R)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        q = np.array([x, y, z, w])
        return q / np.linalg.norm(q)


# ---------- 辅助转换函数 ----------
def quat_wxyz_to_xyzw(q_wxyz):
    """从 [w,x,y,z] 转为 [x,y,z,w]"""
    return np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])


def quat_xyzw_to_wxyz(q_xyzw):
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


def euler_to_quat_xyzw(euler, seq="xyz"):
    return Rotation.from_euler(seq, euler).as_quat()  # scipy returns [x,y,z,w]


def quat_to_euler(q_xyzw, seq="xyz"):
    return Rotation.from_quat(q_xyzw).as_euler(seq, degrees=True)


# 固定旋转：机体系 -> 相机系
R_CB = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float64)
q_CB = Rotation.from_matrix(R_CB).as_quat()  # xyzw


# ---------- 主流程 ----------
def main():
    # 路径设置
    data_root = Path(r"D:\Documents\BIT\2025-2026_2\a2rl\monorace_perception\assets\anim_test-1")
    img_dir = data_root / "imgs"
    meta_path = data_root / "metadata.json"
    imu_path = data_root / "imu_data.csv"
    model_path = Path("models/best.pt")
    output_dir = Path("output/pnp_test_anim1_2")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 读取元数据
    with open(meta_path, "r") as f:
        meta = json.load(f)
    cam = meta["camera"]
    res_x, res_y = cam["resolution_x"], cam["resolution_y"]
    fx = cam["focal_length_mm"] * (res_x / cam["sensor_width_mm"])
    fy = cam["focal_length_mm"] * (res_y / cam["sensor_height_mm"])
    cx = res_x / 2.0
    cy = res_y / 2.0
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    print("camera_matrix: \n" + str(camera_matrix))
    print(cam)
    # 门框世界位姿（来自metadata）
    gate_pos_w = np.array(meta["gate"]["position"], dtype=np.float64)
    gate_euler = np.array(meta["gate"]["euler_xyz_rad"], dtype=np.float64)
    q_gate_w = euler_to_quat_xyzw(gate_euler, seq="xyz")  # xyzw
    # 门框尺寸
    gate_width = meta["gate"]["dimensions"][1]  # 2.7
    gate_height = meta["gate"]["dimensions"][2]  # 2.7

    # 读取IMU数据
    imu_df = pd.read_csv(imu_path)
    # 提取每帧的真实无人机位姿
    frames = imu_df["frame"].values.astype(int)
    times = imu_df["time"].values
    pos_drone_w_true = imu_df[["pos_x", "pos_y", "pos_z"]].values.astype(np.float64)
    # 四元数 w,x,y,z -> xyzw
    q_drone_w_true = np.array(
        [quat_wxyz_to_xyzw(row) for row in imu_df[["quat_w", "quat_x", "quat_y", "quat_z"]].values]
    )

    # 加载YOLO模型
    print(f"Loading YOLO model from {model_path}...")
    yolo = YOLO(str(model_path))

    # 初始化检测器和位姿估计器
    detector = QuAdGate()
    pose_est = PoseEstimator(
        gate_width=gate_width, gate_height=gate_height, camera_matrix=camera_matrix, image_size=(res_x, res_y)
    )

    # 存储结果
    results = []

    for idx in range(len(frames)):
        f = frames[idx]
        img_file = img_dir / f"{f:04d}.jpg"
        if not img_file.exists():
            print(f"Image {img_file} not found, skipping")
            continue
        img_bgr = cv2.imread(str(img_file))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # YOLO推理
        yolo_results = yolo(img_rgb, conf=0.5, iou=0.7, max_det=10, half=True, verbose=False)
        result = yolo_results[0]

        # 提取mask并resize到图像尺寸
        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            if masks.ndim == 2:
                masks = masks[np.newaxis, ...]
            mask = (np.max(masks, axis=0) > 0.5).astype(np.float32)
        else:
            mask = np.zeros((res_y, res_x), dtype=np.float32)
        # 调整mask尺寸到图像尺寸（YOLO输出可能不同）
        if mask.shape[:2] != (res_y, res_x):
            mask = cv2.resize(mask, (res_x, res_y))

        # 角点检测
        detection = detector.detect(mask)
        if detection is None or detection.confidence < 0.3:
            # 无效检测
            gate_pos_c_est = [np.nan, np.nan, np.nan]
            gate_euler_c_est = [np.nan, np.nan, np.nan]
            drone_pos_w_est = [np.nan, np.nan, np.nan]
            drone_euler_w_est = [np.nan, np.nan, np.nan]
        else:
            print(detection.corners)
            # PnP估计门框在相机系下的位姿
            gate_pose = pose_est.estimate_pose(detection)
            if gate_pose is None:
                gate_pos_c_est = [np.nan, np.nan, np.nan]
                gate_euler_c_est = [np.nan, np.nan, np.nan]
                drone_pos_w_est = [np.nan, np.nan, np.nan]
                drone_euler_w_est = [np.nan, np.nan, np.nan]
            else:
                # 估计门框在相机系下的位置和姿态（四元数xyzw）
                pos_gate_c_est = gate_pose.position
                q_gate_c_est = gate_pose.orientation  # xyzw
                gate_euler_c_est = quat_to_euler(q_gate_c_est, seq="xyz")  # deg

                # 反推无人机在世界系下的位姿
                # 姿态：q_WB = q_WG * inv(q_CG) * q_CB
                q_WG = q_gate_w  # xyzw
                q_CG = q_gate_c_est
                # q_CB = q_CB  # xyzw
                # 四元数乘法
                from scipy.spatial.transform import Rotation as R

                # 使用Rotation对象方便运算
                r_WG = R.from_quat(q_WG)
                r_CG = R.from_quat(q_CG)
                r_CB = R.from_quat(q_CB)
                # q_WB = q_WG * inv(q_CG) * q_CB
                r_WB = r_WG * r_CG.inv() * r_CB
                q_WB_est = r_WB.as_quat()  # xyzw
                R_WB_est = r_WB.as_matrix()
                # 位置：pos_drone_w = pos_gate_w - R_WB @ R_BC @ pos_gate_c
                R_BC = R_CB.T  # 相机系->机体系
                pos_drone_w_est = gate_pos_w - R_WB_est @ R_BC @ pos_gate_c_est

                drone_pos_w_est = pos_drone_w_est.tolist()
                drone_euler_w_est = quat_to_euler(q_WB_est, seq="xyz")  # deg

                gate_pos_c_est = pos_gate_c_est.tolist()
                gate_euler_c_est = gate_euler_c_est.tolist()

        # 计算真实门框在相机系下的位姿
        # 无人机真实位姿
        pos_drone_w_true_i = pos_drone_w_true[idx]
        q_drone_w_true_i = q_drone_w_true[idx]  # xyzw
        # 门框世界位姿
        pos_gate_w_i = gate_pos_w
        q_gate_w_i = q_gate_w
        # 计算门框在相机系下
        r_WB_true = Rotation.from_quat(q_drone_w_true_i)  # 机体系->世界
        r_BW_true = r_WB_true.inv()  # 世界->机体系
        r_WG_true = Rotation.from_quat(q_gate_w_i)
        r_CB = Rotation.from_quat(q_CB)
        # 门框在相机系下的旋转：r_CG = r_CB * r_BW * r_WG
        r_CG_true = r_CB * r_BW_true * r_WG_true
        q_gate_c_true = r_CG_true.as_quat()  # xyzw
        gate_euler_c_true = quat_to_euler(q_gate_c_true, seq="xyz")
        # 位置：pos_gate_c = R_CB * R_BW * (pos_gate_w - pos_drone_w)
        pos_gate_c_true = R_CB @ r_BW_true.as_matrix() @ (pos_gate_w_i - pos_drone_w_true_i)
        gate_pos_c_true = pos_gate_c_true.tolist()

        # 真实无人机位姿
        drone_pos_w_true = pos_drone_w_true_i.tolist()
        drone_euler_w_true = quat_to_euler(q_drone_w_true_i, seq="xyz")

        # 存储这一帧的数据
        corners = detection.corners if detection is not None else np.full((4, 2), np.nan)
        results.append(
            {
                "frame": f,
                "gate_pos_c_est_x": gate_pos_c_est[0],
                "gate_pos_c_est_y": gate_pos_c_est[1],
                "gate_pos_c_est_z": gate_pos_c_est[2],
                "gate_euler_c_est_roll": gate_euler_c_est[0],
                "gate_euler_c_est_pitch": gate_euler_c_est[1],
                "gate_euler_c_est_yaw": gate_euler_c_est[2],
                "gate_pos_c_true_x": gate_pos_c_true[0],
                "gate_pos_c_true_y": gate_pos_c_true[1],
                "gate_pos_c_true_z": gate_pos_c_true[2],
                "gate_euler_c_true_roll": gate_euler_c_true[0],
                "gate_euler_c_true_pitch": gate_euler_c_true[1],
                "gate_euler_c_true_yaw": gate_euler_c_true[2],
                "drone_pos_w_est_x": drone_pos_w_est[0],
                "drone_pos_w_est_y": drone_pos_w_est[1],
                "drone_pos_w_est_z": drone_pos_w_est[2],
                "drone_euler_w_est_roll": drone_euler_w_est[0],
                "drone_euler_w_est_pitch": drone_euler_w_est[1],
                "drone_euler_w_est_yaw": drone_euler_w_est[2],
                "drone_pos_w_true_x": drone_pos_w_true[0],
                "drone_pos_w_true_y": drone_pos_w_true[1],
                "drone_pos_w_true_z": drone_pos_w_true[2],
                "drone_euler_w_true_roll": drone_euler_w_true[0],
                "drone_euler_w_true_pitch": drone_euler_w_true[1],
                "drone_euler_w_true_yaw": drone_euler_w_true[2],
                "corner_tl_x": corners[0][0],
                "corner_tl_y": corners[0][1],
                "corner_tr_x": corners[1][0],
                "corner_tr_y": corners[1][1],
                "corner_br_x": corners[2][0],
                "corner_br_y": corners[2][1],
                "corner_bl_x": corners[3][0],
                "corner_bl_y": corners[3][1],
            }
        )

    # 保存CSV
    df = pd.DataFrame(results)
    csv_path = output_dir / "pose_estimation_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")

    # 绘图
    # 设置figsize不超过12，分两个图：门框位姿对比、无人机位姿对比
    fig1, axes1 = plt.subplots(2, 3, figsize=(12, 8))
    fig2, axes2 = plt.subplots(2, 3, figsize=(12, 8))
    fig1.suptitle("Gate Pose in Camera Frame: Estimated vs True")
    fig2.suptitle("Drone Pose in World Frame: Estimated vs True")

    frames_vals = df["frame"].values

    # 门框位置
    pos_cols = [
        ("gate_pos_c_est_x", "gate_pos_c_true_x"),
        ("gate_pos_c_est_y", "gate_pos_c_true_y"),
        ("gate_pos_c_est_z", "gate_pos_c_true_z"),
    ]
    for i, (est, true) in enumerate(pos_cols):
        ax = axes1[0, i]
        ax.plot(frames_vals, df[est], "bo-", label="Est", markersize=3)
        ax.plot(frames_vals, df[true], "r^-", label="True", markersize=3)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Position (m)")
        ax.set_title(f'Axis {["X","Y","Z"][i]}')
        ax.legend()
        ax.grid(True)

    # 门框姿态
    euler_cols = [
        ("gate_euler_c_est_roll", "gate_euler_c_true_roll"),
        ("gate_euler_c_est_pitch", "gate_euler_c_true_pitch"),
        ("gate_euler_c_est_yaw", "gate_euler_c_true_yaw"),
    ]
    for i, (est, true) in enumerate(euler_cols):
        ax = axes1[1, i]
        ax.plot(frames_vals, df[est], "bo-", label="Est", markersize=3)
        ax.plot(frames_vals, df[true], "r^-", label="True", markersize=3)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Euler angle (deg)")
        ax.set_title(f'Axis {["Roll","Pitch","Yaw"][i]}')
        ax.legend()
        ax.grid(True)

    # 无人机位置
    pos_cols2 = [
        ("drone_pos_w_est_x", "drone_pos_w_true_x"),
        ("drone_pos_w_est_y", "drone_pos_w_true_y"),
        ("drone_pos_w_est_z", "drone_pos_w_true_z"),
    ]
    for i, (est, true) in enumerate(pos_cols2):
        ax = axes2[0, i]
        ax.plot(frames_vals, df[est], "bo-", label="Est", markersize=3)
        ax.plot(frames_vals, df[true], "r^-", label="True", markersize=3)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Position (m)")
        ax.set_title(f'Axis {["X","Y","Z"][i]}')
        ax.legend()
        ax.grid(True)

    # 无人机姿态
    euler_cols2 = [
        ("drone_euler_w_est_roll", "drone_euler_w_true_roll"),
        ("drone_euler_w_est_pitch", "drone_euler_w_true_pitch"),
        ("drone_euler_w_est_yaw", "drone_euler_w_true_yaw"),
    ]
    for i, (est, true) in enumerate(euler_cols2):
        ax = axes2[1, i]
        ax.plot(frames_vals, df[est], "bo-", label="Est", markersize=3)
        ax.plot(frames_vals, df[true], "r^-", label="True", markersize=3)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Euler angle (deg)")
        ax.set_title(f'Axis {["Roll","Pitch","Yaw"][i]}')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    fig1.savefig(output_dir / "gate_pose_comparison.png", dpi=150)
    fig2.savefig(output_dir / "drone_pose_comparison.png", dpi=150)
    print(f"Figures saved to {output_dir}")

    plt.show()


if __name__ == "__main__":
    main()
