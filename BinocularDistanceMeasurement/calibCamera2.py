import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt


def calibCamera(images_left_dir, images_right_dir, chessboard_size=(9, 6), square_size=24, showMatch=False):

    # 准备棋盘格点的世界坐标（假设棋盘格在z=0的平面上）
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    # 用于存储所有图像的棋盘格点
    objpoints = []
    imgpoints_left = []
    imgpoints_right = []

    # 读取左相机和右相机的图像
    images_left = glob.glob(images_left_dir)
    images_right = glob.glob(images_right_dir)

    # 确保左右图像数量匹配
    if len(images_left) != len(images_right):
        raise ValueError("左右相机图像数量不匹配")

    img_left_list = []
    img_right_list = []
    for img_left, img_right in zip(images_left, images_right):
        # 读取图像
        imgL = cv2.imread(img_left)
        imgR = cv2.imread(img_right)
        imgL,imgR = adjustBrightnessToSameLevel(imgL,imgR)

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
            img_right_list.append(img_right)
            img_left_list.append(img_left)
        else:
            print(f"Skipping pair: {img_left}, {img_right} due to failed corner detection")

    # 确保有足够的图像点进行标定
    if len(imgpoints_left) < 1 or len(imgpoints_right) < 1:
        raise ValueError("没有足够的有效图像进行标定")

    print("有效图像对数量:", len(imgpoints_left))

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
    # 计算重投影误差
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecsL[i], tvecsL[i], mtxL, distL)
        error = cv2.norm(imgpoints_left[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error

    mean_error = total_error / len(objpoints)
    print(f"Left camera mean re-projection error: {mean_error}")

    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecsR[i], tvecsR[i], mtxR, distR)
        error = cv2.norm(imgpoints_right[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error

    mean_error = total_error / len(objpoints)
    print(f"Right camera mean re-projection error: {mean_error}")


    """
    查看校准结果
    """

    if showMatch:
        # 绘制重投影点
        def draw_reprojected_points(image, corners, reprojected_corners):
            for i in range(len(corners)):
                actual_corner = tuple(map(int, corners[i].ravel()))
                reproj_corner = tuple(map(int, reprojected_corners[i].ravel()))
                cv2.circle(image, actual_corner, 5, (0, 0, 255), -1)  # 红色圆圈表示实际角点
                cv2.circle(image, reproj_corner, 3, (0, 255, 0), -1)  # 绿色圆圈表示重投影角点
            return image
        match_num = len(img_right_list)


        #print(img_left_list)

        for i in range(match_num):
            plt.figure(figsize=(8, 4))
            imgL = cv2.imread(img_left_list[i])
            imgR = cv2.imread(img_right_list[i])

            reprojected_cornersL, _ = cv2.projectPoints(objpoints[i], rvecsL[i], tvecsL[i], mtxL, distL)
            reprojected_cornersR, _ = cv2.projectPoints(objpoints[i], rvecsR[i], tvecsR[i], mtxR, distR)

            imgL = draw_reprojected_points(imgL, imgpoints_left[i], reprojected_cornersL)
            imgR = draw_reprojected_points(imgR, imgpoints_right[i], reprojected_cornersR)
            plt.subplot(1, 2, 1)
            plt.title(f'Left Image-{i}')
            plt.imshow(cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB))
            plt.subplot(1,2, 2)
            plt.title(f'Right Image-{i}')
            plt.imshow(cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB))
            plt.tight_layout()
            plt.show()

    # 输出校准结果
    print("Left camera matrix:\n", mtxL)
    print("Left distortion coefficients:\n", distL)
    print("Right camera matrix:\n", mtxR)
    print("Right distortion coefficients:\n", distR)
    print("Rotation matrix between cameras:\n", R)
    print("Translation vector between cameras:\n", T)
    return mtxL, distL, mtxR, distR, R, T

def adjustBrightnessToSameLevel(img1,img2):
    # 读取两个灰度图像


    # 计算每个图像的平均亮度
    mean1 = np.mean(img1)
    mean2 = np.mean(img2)

    # 根据平均亮度进行图像归一化
    img1_norm = cv2.normalize(img1, None, alpha=mean2, beta=0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    img2_norm = img2
    return img1_norm,img2_norm
