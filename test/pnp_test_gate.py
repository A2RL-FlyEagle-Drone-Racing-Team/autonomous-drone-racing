"""
给出下列python代码：
import numpy as np
import cv2
from scipy.spatial.transform import Rotation

# 相机传感器尺寸（宽 & 高，mm）
cam_S = 36
# 焦距（mm）
f = 75
# 机体在世界系中姿态欧拉角（偏航Z-俯仰Y-滚转X，deg）
cam_euler = [30, 20, 0]
# 机体和相机在世界系中坐标（XYZ，m）
cam_pos = [-15, -9, 6]

# 图像高度（像素）
h = 128
# 图像宽度（像素）
w = 128

# 门框角点相对于门框中心坐标
points_3d = np.array([
    [0, 2, 2],
    [0, -2, 2],
    [0, -2, -2],
    [0, 2, -2]
], dtype=np.float32
)
# 门框角点在图像中的像素坐标
points_2d = np.array([
    [33, 25],
    [89, 33],
    [86, 93],
    [35, 80]
], dtype=np.float32
)
已知相机坐标系xyz轴正向分别为右、下、前，机体坐标系和xyz轴正向分别为前、左、上，
相机系和机体系原点相互重合，且相机系z轴正方向与机体系x轴正方向一致；
相机画面无畸变；门框中心世界坐标和姿态角未知。
完成代码的后续部分，完成以下任务：
1. 求出相机的内参矩阵和外参矩阵（反映相机系和机体系之间的变换）
2. 使用cv2.solvePnPRansac进行PnP求解
3. 求出门框分别在机体系和世界系下的中心坐标和姿态角
"""

import numpy as np
import cv2
from scipy.spatial.transform import Rotation

# ==================== 已知参数（已提供）====================
cam_S = 36
f = 75
cam_euler = [30, 20, 0]   # 机体在世界系中的欧拉角 (偏航Z, 俯仰Y, 滚转X) deg
cam_pos = [-15, -9, 6]    # 机体/相机在世界系中的位置 (X,Y,Z) m
h, w = 128, 128
points_3d = np.array([
    [0,  2,  2],
    [0, -2,  2],
    [0, -2, -2],
    [0,  2, -2]
], dtype=np.float32)
points_2d = np.array([
    [33, 25],
    [89, 33],
    [86, 93],
    [35, 80]
], dtype=np.float32)

# ==================== 任务1：内参矩阵 & 相机-机体系外参 ====================
# 内参矩阵
fx = fy = f * (w / cam_S)   # 像素焦距
cx, cy = w / 2.0, h / 2.0
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]], dtype=np.float32)

# 相机系与机体系之间的旋转矩阵 (机体 → 相机)
# 定义：相机系 Xc→右, Yc→下, Zc→前；机体系 Xb→前, Yb→左, Zb→上
R_body_to_cam = np.array([
    [0, -1,  0],
    [0,  0, -1],
    [1,  0,  0]
], dtype=np.float32)

print("="*50)
print("任务1：相机内参矩阵 K：")
print(K)
print("\n相机系与机体系之间的旋转矩阵 R_body_to_cam (机体→相机)：")
print(R_body_to_cam)
print("平移向量为零（原点重合）")

# ==================== 任务2：PnP 求解（门框局部系 → 相机系）====================
success, rvec, tvec, inliers = cv2.solvePnPRansac(
    points_3d, points_2d, K, None,
    iterationsCount=100, reprojectionError=1.0,
    flags=cv2.SOLVEPNP_ITERATIVE
)

# 旋转向量 → 旋转矩阵
R_door_to_cam, _ = cv2.Rodrigues(rvec)
print("\n" + "="*50)
print("任务2：PnP 求解结果")
print("门框局部系 → 相机系 旋转矩阵 R_door_to_cam：")
print(R_door_to_cam)
print("门框中心在相机系中的坐标 (tvec)：")
print(tvec.ravel())

# ==================== 任务3：求门框在机体系和世界系下的位姿 ====================
# 3.1 已知的机体在世界系中的位姿
# 机体 → 世界 旋转矩阵 (欧拉角 ZYX 内旋)
R_body_to_world = Rotation.from_euler('zyx', cam_euler, degrees=True).as_matrix()
t_body_in_world = np.array(cam_pos).reshape(3, 1)   # 机体原点在世界系中的位置

# 相机 → 世界 旋转矩阵 (相机与机体原点重合，但姿态不同)
R_cam_to_world = R_body_to_world @ R_body_to_cam.T   # 因为 p_cam = R_body_to_cam * p_body
t_cam_in_world = t_body_in_world.copy()              # 位置相同

# 3.2 门框中心在机体系中的坐标
# 门框中心在相机系中为 tvec，转换到机体系：p_body = R_cam_to_body * tvec
R_cam_to_body = R_body_to_cam.T
door_center_in_body = (R_cam_to_body @ tvec).ravel()
print("\n" + "="*50)
print("任务3：门框中心在机体系中的坐标 (前, 左, 上)：")
print(door_center_in_body)

# 3.3 门框中心在世界系中的坐标
# p_world = t_cam_in_world + R_cam_to_world @ tvec
door_center_in_world = (t_cam_in_world + R_cam_to_world @ tvec).ravel()
print("\n门框中心在世界系中的坐标 (X, Y, Z)：")
print(door_center_in_world)

# 3.4 门框在机体系下的姿态 (局部系 → 机体)
R_door_to_body = R_cam_to_body @ R_door_to_cam
euler_door_in_body = Rotation.from_matrix(R_door_to_body).as_euler('zyx', degrees=True)
print("\n门框在机体系下的姿态角 (偏航Z, 俯仰Y, 滚转X) [deg]：")
print(euler_door_in_body)

# 3.5 门框在世界系下的姿态 (局部系 → 世界)
R_door_to_world = R_cam_to_world @ R_door_to_cam
euler_door_in_world = Rotation.from_matrix(R_door_to_world).as_euler('zyx', degrees=True)
print("\n门框在世界系下的姿态角 (偏航Z, 俯仰Y, 滚转X) [deg]：")
print(euler_door_in_world)