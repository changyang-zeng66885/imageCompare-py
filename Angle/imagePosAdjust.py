import math

import cv2
import numpy as np
import matplotlib.pyplot as plt


# 给定两张图片，以及在图片1中指定的关键点，返回两张图片关键点的对应关系。
def manualKeyPoints(img1,img2,specified_keypoints_coords,distance_threshold = 100,kpmax = 1000):
    # 将图片转换为灰度图
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 初始化SIFT特征检测器
    sift = cv2.SIFT_create()

    # 搜索 specified_keypoints_coords 附近的关键点坐标
    kpn1, desn1 = sift.detectAndCompute(gray1, None)
    specified_keypoints_coords = np.array(specified_keypoints_coords)
    # 初始化新的关键点和描述符列表
    kp1 = []
    des1 = []
    # 设置距离阈值
    for specPoint in specified_keypoints_coords:
        kpcnt = 0
        xs,ys = specPoint
        for i, kp in enumerate(kpn1):
            x, y = kp.pt
            if (xs-x)**2 + (ys-y)**2 < distance_threshold**2 and kpcnt <= kpmax :
                kp1.append(kp)
                des1.append(desn1[i])
                kpcnt += 1

    # 将描述符转换为numpy数组
    des1 = np.array(des1)
    print(f"KeyPoints 1 number:{len(kp1)}")

    # 检测图片2的关键点和计算描述子
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # 将描述子转换为浮点型
    # 由于cv2.FlannBasedMatcher不支持ORB描述子(descriptors)的默认格式。FLANN匹配器通常用于SIFT和SURF描述子，它们是浮点型的，而ORB描述子是8位的整数型。要解决这个问题，可以将ORB描述子转换为浮点型。
    des1 = des1.astype('float32')
    des2 = des2.astype('float32')

    # 使用FLANN匹配器进行匹配
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # 使用Lowe's ratio test来筛选匹配点
    good_matches = []
    for m,n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    print(f"Good match number:{len(good_matches)}")

    # 找到照片2中对应的关键点位置
    corresponding_points = []
    for match in good_matches:
        corresponding_points.append((kp1[match.queryIdx].pt, kp2[match.trainIdx].pt))

    srcPts = []
    dstPts = []

    # 在照片2中标注关键点和位移箭头
    img2_with_keypoints = img2.copy()
    for i, (pt1, pt2) in enumerate(corresponding_points):
        start_point = (int(pt1[0]), int(pt1[1]))
        end_point = (int(pt2[0]), int(pt2[1]))
        srcPts.append(start_point)
        dstPts.append(end_point)
        cv2.circle(img2_with_keypoints, start_point , 20, (0, 0, 255), -1)
        cv2.circle(img2_with_keypoints, end_point, 20, (0, 255, 0), -1)
        cv2.arrowedLine(img2_with_keypoints, start_point, end_point, (255, 0, 0), 5, tipLength=0.05)
    srcPts = np.array(srcPts, dtype=np.float32).reshape(-1, 2)
    dstPts = np.array(dstPts, dtype=np.float32).reshape(-1, 2)

    return srcPts,dstPts,kp1,kp2

def showPointMatchWithArrowLine(srcPts,dstPts,img2,arrowLabelList = []):
    if not len(srcPts) == len(dstPts):
        print(f"len(srcPts)({len(srcPts)}) != len(dstPts)({len(dstPts)})!")
        return cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    img2_with_keypoints = img2.copy()
    for i in range(len(srcPts)):
        start_point = (int(srcPts[i][0]),int(srcPts[i][1]))
        end_point = (int(dstPts[i][0]),int(dstPts[i][1]))
        cv2.circle(img2_with_keypoints, start_point, 5, (0, 0, 255), -1)
        cv2.circle(img2_with_keypoints, end_point, 5, (0, 255, 0), -1)
        cv2.arrowedLine(img2_with_keypoints, start_point, end_point, (255, 0, 0), 2, tipLength=0.05)
        if len(arrowLabelList) == len(srcPts):
            cv2.putText(img2_with_keypoints, f'{arrowLabelList[i] :.4f}m', ((end_point[0]+start_point[0])//2+15,(end_point[1]+start_point[1])//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
            #print(f'{arrowLabelList[i] } m')

    return cv2.cvtColor(img2_with_keypoints, cv2.COLOR_BGR2RGB)


# 绘制关键点（指定关键点或者自动识别的关键点）
def showkpInImage(kp1,img1,specified_keypoints_coords = [],color = (0,0,255)):
    img1_with_keypoints = cv2.drawKeypoints(img1, kp1, None, color=(0, 255, 0),flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
    for coord in specified_keypoints_coords:
        cv2.circle(img1_with_keypoints, coord, 5, color, -1)  # 红色 (BGR), 半径为10, 实心圆
    return img1_with_keypoints


# 根据关键点的对应关系（srcPts，dstPts），将两张图片调整到一样的位置
def adjustImagesToSamePosWithSrcDst(image1,image2,srcPts,dstPts):
    # 计算变换矩阵
    M, mask = cv2.findHomography(dstPts, srcPts, cv2.RANSAC, 5.0)

    # 应用变换矩阵
    h, w, d = image1.shape
    image2_aligned = cv2.warpPerspective(image2, M, (w, h))





 
    # 找到有效区域
    # Grayimg2 = cv2.cvtColor(image2_aligned, cv2.COLOR_BGR2GRAY)  # 先要转换为灰度图片
    # ret, thresh = cv2.threshold(Grayimg2, 1, 255, cv2.THRESH_BINARY)  # 这里的第二个参数要调，是阈值！
    # contours, _ = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    #
    # x, y, w, h = cv2.boundingRect(contours[0])
    #
    # # 裁剪图像
    # image1_cropped = image1[y:y + h, x:x + w]
    # image2_aligned_cropped = image2_aligned[y:y + h, x:x + w]

    # 显示结果
    # plt.figure(figsize=(2*4,1*4))
    # plt.title("Image Adjust Result")
    # plt.subplot(121), plt.imshow(image1), plt.title("image1")
    # plt.subplot(122), plt.imshow(image2_aligned_cropped), plt.title("image2_aligned_cropped")
    # plt.show()
    # return image1_cropped,image2_aligned_cropped
    return image1,image2_aligned


# 使用最小二乘法拟合直线
def fit_line(points):
    x = points[:, 0]
    y = points[:, 1]
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]

    # # 绘制点和拟合直线
    # plt.scatter(x, y, color='red', label='Data Points')
    # plt.plot(x, m * x + c, color='blue', label='Fitted Line')
    # # 设置图例和标签
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.title('Linear Fit using Least Squares')
    # plt.legend()
    # plt.grid(True)
    # # 展示图形
    # plt.show()

    return m, c

# 绘制拟合直线
def draw_line(image, m, c, color):
    # print(f"shape0:{image.shape[0]}, shape1: {image.shape[1]}")
    y1 = int(m * 0 + c)
    y2 = int(m * image.shape[1] + c)
    cv2.line(image, (0, y1), (image.shape[1], y2), color, 2)

def transferPointsByM(srcPts_t,M):
    print("srcPts = ",srcPts_t)
    print("M = ",M)
    # 将点集扩展为三维坐标（添加第三个维度为1）
    srcPts_t_homogeneous = np.concatenate([srcPts_t, np.ones((srcPts_t.shape[0], 1))], axis=1)
    # 进行矩阵乘法，应用变换矩阵 M
    dstPts_t_homogeneous = np.dot(M, srcPts_t_homogeneous.T).T
    # 归一化，转换为二维坐标
    dstPts_t = dstPts_t_homogeneous[:, :2] / dstPts_t_homogeneous[:, 2, np.newaxis]
    return dstPts_t


# getDistanceFromStandardImage
def getDistanceFromStandardImage(basePts,standardPts,srcPts,dstPts,stdPix2ActDistanceRate):
    print("BasePts = ",basePts)
    print("standardPts = ", standardPts)

    MBase, mask = cv2.findHomography(basePts,standardPts,  cv2.RANSAC, 5.0)
    #MBase, mask = cv2.findHomography(standardPts, standardPts, cv2.RANSAC, 5.0)
    print("MBase = ",MBase)

    srcPts_std = transferPointsByM(srcPts,MBase)
    dstPts_std = transferPointsByM(dstPts, MBase)

    actualDistanceList = []
    for i in range(len(srcPts_std)):
        x1,y1 = srcPts_std[i][0],srcPts_std[i][1]
        x2, y2 = dstPts_std[i][0], dstPts_std[i][1]
        print(f"src ({x1},{y1}) -> dst ({x2},{y2})")
        actDis = math.sqrt( (x1-x2)**2 + (y1-y2)**2)
        actualDistanceList.append(actDis*stdPix2ActDistanceRate)
    return actualDistanceList



