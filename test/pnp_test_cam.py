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
已知相机坐标系xyz轴正向分别为右、下、前，机体坐标系和xyz轴正向分别为前、左、上，相机系和机体系原点相互重合，
且相机系z轴正方向与机体系x轴正方向一致；相机画面无畸变；
门框中心位于世界坐标系原点，姿态角均为0，门框局部坐标系x轴正向与世界系x轴正向一致。
完成代码的后续部分，完成以下任务：
1. 求出相机的内参矩阵和外参矩阵（反映相机系和机体系之间的变换）
2. 使用cv2.solvePnPRansac进行PnP求解
3. 利用PnP结果求出机体在世界坐标系中的位置和姿态，并与已知条件中给出的机体位置、姿态进行比较
"""

import numpy as np
import cv2
from scipy.spatial.transform import Rotation

# ==================== 已知参数（已提供）====================
cam_S = 36          # 传感器宽度与高度（假设正方形）mm
f = 75              # 焦距 mm
cam_euler = [30, 20, 0]   # 机体在世界系中的欧拉角 (偏航Z, 俯仰Y, 滚转X) deg
cam_pos = [-15, -9, 6]    # 机体/相机在世界系中的位置 (X,Y,Z) m
h, w = 128, 128           # 图像高、宽 像素

# 门框角点在世界坐标系中的坐标（门框中心位于原点，无旋转）
points_3d = np.array([
    [0,  2,  2],
    [0, -2,  2],
    [0, -2, -2],
    [0,  2, -2]
], dtype=np.float32)

# 对应角点在图像中的像素坐标
points_2d = np.array([
    [33, 25],
    [89, 33],
    [86, 93],
    [35, 80]
], dtype=np.float32)

# ==================== 任务1：内参矩阵与外参矩阵 ====================
# 内参矩阵 K
fx = fy = f * (w / cam_S)   # 假设传感器宽高均为 cam_S mm
cx, cy = w / 2.0, h / 2.0
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]], dtype=np.float32)

# 相机系与机体系之间的旋转矩阵（外参）
# 定义：R_body_to_cam 将机体坐标系下的点转换到相机坐标系
# 根据题意：相机系 Zc ← 机体系 Xb（前）；相机系 Xc ← 机体系 -Yb（左的负方向）；相机系 Yc ← 机体系 -Zb（上的负方向）
R_body_to_cam = np.array([
    [0, -1,  0],
    [0,  0, -1],
    [1,  0,  0]
], dtype=np.float32)

print("内参矩阵 K:\n", K)
print("\n相机系与机体系之间的旋转矩阵 R_body_to_cam:\n", R_body_to_cam)
print("（说明：将机体坐标系中的点转换到相机坐标系）")

# ==================== 任务2：使用 cv2.solvePnPRansac 求解 ====================
# 注意：points_3d 是世界坐标系下的3D点，points_2d 是对应的像素坐标
# 无畸变，distCoeffs=None
success, rvec, tvec, inliers = cv2.solvePnPRansac(points_3d, points_2d, K, None,
                                         iterationsCount=100, reprojectionError=1.0,
                                         flags=cv2.SOLVEPNP_ITERATIVE)
assert success

# 将旋转向量转换为旋转矩阵
R_world_to_cam, _ = cv2.Rodrigues(rvec)
print("\nPnP 求解结果：")
print("旋转向量 rvec (世界→相机):\n", rvec.ravel())
print("平移向量 tvec (世界原点在相机系中的坐标):\n", tvec.ravel())

# ==================== 任务3：利用PnP结果计算机体位姿并与已知值比较 ====================
# 相机在世界坐标系中的姿态（旋转矩阵）和位置
R_cam_to_world = R_world_to_cam.T          # 相机系 → 世界系
t_cam_in_world = -R_cam_to_world @ tvec    # 相机（也是机体）在世界系中的位置

# 机体在世界坐标系中的姿态
R_body_to_world = R_cam_to_world @ R_body_to_cam

# 从旋转矩阵提取欧拉角（ZYX 顺序，与输入 cam_euler 一致）
euler_body = Rotation.from_matrix(R_body_to_world).as_euler('zyx', degrees=True)

print("\n===== 计算得到的机体在世界系中的位姿 =====")
print(f"机体位置 (X, Y, Z): {t_cam_in_world.ravel()} m")
print(f"机体姿态 (偏航Z, 俯仰Y, 滚转X): {euler_body} deg")

print("\n===== 已知条件中的机体位姿 =====")
print(f"机体位置 (X, Y, Z): {cam_pos} m")
print(f"机体姿态 (偏航Z, 俯仰Y, 滚转X): {cam_euler} deg")

# 比较误差
pos_error = np.linalg.norm(t_cam_in_world.ravel() - np.array(cam_pos))
euler_error = np.linalg.norm(euler_body - np.array(cam_euler))
print("\n===== 位姿误差 =====")
print(f"位置误差 (欧氏距离): {pos_error:.6f} m")
print(f"姿态误差 (欧拉角范数): {euler_error:.6f} deg")

# 可选：验证重投影误差
proj_points, _ = cv2.projectPoints(points_3d, rvec, tvec, K, None)
print("\n重投影误差（每个角点）:")
for i, (p_orig, p_proj) in enumerate(zip(points_2d, proj_points.squeeze())):
    err = np.linalg.norm(p_orig - p_proj)
    print(f"  点{i+1}: {err:.4f} 像素")