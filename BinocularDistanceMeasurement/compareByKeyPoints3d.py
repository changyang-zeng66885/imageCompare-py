import numpy as np
import cv2
"""
要实现一个简单的双目视觉深度估计和3D坐标计算，你需要完成以下几个步骤：
相机校准：获得相机的内参和外参矩阵。
立体对齐：对两幅图像进行校正，使得它们在同一平面上。
视差图计算：计算两幅图像的视差图。
深度图生成：利用视差图生成深度图。
3D坐标计算：根据深度图和相机参数计算图像中点的3D坐标。
以下是一个简单的Python代码示例，展示如何使用OpenCV实现这些步骤：
"""

# 加载左右图像
left_img = cv2.imread('left_image.jpg', cv2.IMREAD_GRAYSCALE)
right_img = cv2.imread('right_image.jpg', cv2.IMREAD_GRAYSCALE)

# 校正过的相机内参和外参矩阵（假设已经获得这些参数）
# 你需要根据你的相机校准结果来设置这些值
K1 = np.array([[fx1, 0, cx1],
               [0, fy1, cy1],
               [0, 0, 1]])
K2 = np.array([[fx2, 0, cx2],
               [0, fy2, cy2],
               [0, 0, 1]])
D1 = np.array([k1, k2, p1, p2, k3])
D2 = np.array([k1, k2, p1, p2, k3])
R = np.eye(3) # 如果两个相机平行
T = np.array([T1, T2, T3]) # 两个相机之间的平移向量

# 立体校正
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K1, D1, K2, D2, left_img.shape[::-1], R, T)
map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, left_img.shape[::-1], cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, right_img.shape[::-1], cv2.CV_32FC1)

rectified_left = cv2.remap(left_img, map1x, map1y, cv2.INTER_LINEAR)
rectified_right = cv2.remap(right_img, map2x, map2y, cv2.INTER_LINEAR)

# 计算视差图
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(rectified_left, rectified_right)

# 根据视差图生成深度图
depth_map = cv2.reprojectImageTo3D(disparity, Q)

# 计算图像上某一点的3D坐标 (假设点坐标为(x, y))
x, y = 100, 100  # 替换为你要计算的点坐标
point_3d = depth_map[y, x]

print(f"3D coordinates of the point: {point_3d}")

# 显示结果
cv2.imshow('Left Image', left_img)
cv2.imshow('Right Image', right_img)
cv2.imshow('Rectified Left', rectified_left)
cv2.imshow('Rectified Right', rectified_right)
cv2.imshow('Disparity', (disparity / 16.0).astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()