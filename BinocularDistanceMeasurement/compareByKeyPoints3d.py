import numpy as np
import cv2
import matplotlib.pyplot as plt
import calibCamera2
import glob

"""
要实现一个简单的双目视觉深度估计和3D坐标计算，你需要完成以下几个步骤：
相机校准：获得相机的内参和外参矩阵。
立体对齐：对两幅图像进行校正，使得它们在同一平面上。
视差图计算：计算两幅图像的视差图。
深度图生成：利用视差图生成深度图。
3D坐标计算：根据深度图和相机参数计算图像中点的3D坐标。
以下是一个简单的Python代码示例，展示如何使用OpenCV实现这些步骤：
"""


def detect_distance(left_img, right_img, K1, D1, K2, D2, R, T):
    # 立体校正
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K1, D1, K2, D2, left_img.shape[::-1], R, T)
    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, left_img.shape[::-1], cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, right_img.shape[::-1], cv2.CV_32FC1)
    rectified_left = cv2.remap(left_img, map1x, map1y, cv2.INTER_LINEAR)
    rectified_right = cv2.remap(right_img, map2x, map2y, cv2.INTER_LINEAR)

    # 计算视差图
    window_size = 5
    min_disp = 0
    num_disp = 16 * 6  # 必须是16的倍数
    stereo = cv2.StereoSGBM_create(  # 创建SGBM对象
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=window_size,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=20,
        speckleWindowSize=100,
        speckleRange=32
    )
    disparity_map = stereo.compute(rectified_left, rectified_right) # 视差图
    # 将视差图转换为深度图
    depth_map = cv2.convertScaleAbs(disparity_map, alpha=255 / disparity_map.max())

    # 计算图像上某一点的3D坐标 (假设点坐标为(x, y))
    point_cloud = cv2.reprojectImageTo3D(disparity_map, Q)
    # x, y = 100, 100  # 替换为你要计算的点坐标
    # point_3d = point_cloud[y, x]
    # print(f"({x},{y})in image ->3D coordinates of the point: {point_3d}")

    # 显示结果
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title('rectified_left')
    plt.imshow(cv2.cvtColor(rectified_left, cv2.COLOR_BGR2RGB))

    plt.subplot(1, 3, 2)
    plt.title('rectified_right')
    plt.imshow(cv2.cvtColor(rectified_right, cv2.COLOR_BGR2RGB))

    print(depth_map.shape)
    plt.subplot(1, 3, 3)
    plt.imshow(depth_map)
    plt.title('depth_map')

    plt.colorbar()
    plt.show()


# 读取左相机和右相机的图像
imgpoints_left = []
imgpoints_right = []
images_left = glob.glob("demoImage5/calib/left/*.jpg")
images_right = glob.glob('demoImage5/calib/right/*.jpg')

# 校正过的相机内参和外参矩阵（假设已经获得这些参数）
mtxL, distL, mtxR, distR, R1, T1 = calibCamera2.calibCamera('demoImage4/calib/left/*.jpg',
                                                            'demoImage4/calib/right/*.jpg', showMatch=False)
K1 = np.array(mtxL)  # 左相机内参
K2 = np.array(mtxR)  # 右相机内参
D1 = np.array(distL)  # 左相机畸变系数:[k1, k2, p1, p2, k3]
D2 = np.array(distR)  # 又相机畸变系数:[k1, k2, p1, p2, k3]
R = np.array(R1)  # 两个相机的旋转矩阵
T = np.array(T1)  # 两个相机之间的平移向量
#

for img_left, img_right in zip(images_left, images_right):
    print(f"img_left:{img_left}, img_right:{img_right}")
    left_img = cv2.imread(img_left, cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread(img_right, cv2.IMREAD_GRAYSCALE)
    detect_distance(left_img, right_img, K1, D1, K2, D2, R, T)
