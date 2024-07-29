import numpy as np
import cv2
import glob

"""
相机校准是立体视觉系统中的重要步骤，用于获取相机的内参和外参矩阵。这个过程通常包括以下步骤：

拍摄标定图像：使用标定板（通常是棋盘格图案）拍摄多张图像。
检测标定板角点：在每张图像中检测棋盘格图案的角点。
校准单个相机：通过这些角点来校准每个相机，得到内参和畸变系数。
立体校准：使用两个相机的图像对进行立体校准，得到两个相机之间的旋转矩阵和平移向量。
以下是详细的Python代码示例，展示如何使用OpenCV进行这些步骤：

1. 拍摄标定图像
首先，你需要准备一个棋盘格标定板，并从不同角度拍摄多张包含标定板的图像。

2. 检测标定板角点
使用OpenCV的cv2.findChessboardCorners函数检测角点。

3. 校准单个相机
使用cv2.calibrateCamera函数进行单个相机的校准。

4. 立体校准
使用cv2.stereoCalibrate函数进行立体校准。

"""

# 设置棋盘格尺寸
chessboard_size = (9, 6)
square_size = 25  # 单位为毫米

# 准备棋盘格点的世界坐标（假设棋盘格在z=0的平面上）
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# 用于存储所有图像的棋盘格点
objpoints = []
imgpoints_left = []
imgpoints_right = []

# 读取左相机和右相机的图像
images_left = glob.glob('left/*.jpg')
images_right = glob.glob('right/*.jpg')

for img_left, img_right in zip(images_left, images_right):
    # 读取图像
    imgL = cv2.imread(img_left)
    imgR = cv2.imread(img_right)
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    # 找到棋盘格角点
    retL, cornersL = cv2.findChessboardCorners(grayL, chessboard_size, None)
    retR, cornersR = cv2.findChessboardCorners(grayR, chessboard_size, None)

    if retL and retR:
        objpoints.append(objp)

        # 精细化角点位置
        cornersL = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1),
                                    (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001))
        cornersR = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1),
                                    (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001))

        imgpoints_left.append(cornersL)
        imgpoints_right.append(cornersR)

# 校准左相机
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpoints_left, grayL.shape[::-1], None, None)

# 校准右相机
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpoints_right, grayR.shape[::-1], None, None)

# 立体校准
flags = cv2.CALIB_FIX_INTRINSIC
criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 1e-6)

ret, mtxL, distL, mtxR, distR, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_right,
    mtxL, distL, mtxR, distR,
    grayL.shape[::-1], criteria=criteria, flags=flags
)

# 输出校准结果
print("Left camera matrix:\n", mtxL)
print("Left distortion coefficients:\n", distL)
print("Right camera matrix:\n", mtxR)
print("Right distortion coefficients:\n", distR)
print("Rotation matrix between cameras:\n", R)
print("Translation vector between cameras:\n", T)