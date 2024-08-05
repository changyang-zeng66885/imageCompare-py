import cv2
import numpy as np
import matplotlib.pyplot as plt


# 获取照片1中指定关键点在照片2中的对应位置
def find_corresponding_points(kp1, kp2, matches):
    points = []
    for match in matches:
        points.append((kp1[match.queryIdx].pt, kp2[match.trainIdx].pt))
    return points


# 给定两张图片，以及指定的关键点，返回两张图片关键点的对应关系。
def manualKeyPoints(img1,img2,specified_keypoints_coords):
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
    distance_threshold = 100
    # 遍历每一个关键点
    for i, kp in enumerate(kpn1):
        x, y = kp.pt
        distances = np.sqrt(np.sum((specified_keypoints_coords - np.array([x, y])) ** 2, axis=1))
        # 检查是否有任意一个距离小于阈值
        if np.any(distances < distance_threshold):
            kp1.append(kp)
            des1.append(desn1[i])
    # 将描述符转换为numpy数组
    des1 = np.array(des1)

    print(f"KeyPoints 1 number:{len(kp1)}")

    plt.figure(figsize=(2*4,2*4))

    img1_with_keypoints = cv2.drawKeypoints(img1, kp1, None, color=(0, 255, 0), flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
    for coord in specified_keypoints_coords:
        cv2.circle(img1_with_keypoints, coord, 30, (0, 0, 255), -1)  # 红色 (BGR), 半径为10, 实心圆
    plt.subplot(2,2,1),plt.imshow(cv2.cvtColor(img1_with_keypoints, cv2.COLOR_BGR2RGB)),plt.title("img1 with initial target(Red) and keypoints(green)")

    # 检测图片2的关键点和计算描述子
    kp2, des2 = sift.detectAndCompute(gray2, None)

    img2_with_keypoints = cv2.drawKeypoints(img2, kp2, None, color=(0, 255, 0), flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
    plt.subplot(2,2,3),plt.imshow(cv2.cvtColor(img2_with_keypoints, cv2.COLOR_BGR2RGB)),plt.title("img2 with keypoints(green)")

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
    corresponding_points = find_corresponding_points(kp1, kp2, good_matches)

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

    plt.subplot(2,2,4),plt.imshow(cv2.cvtColor(img2_with_keypoints, cv2.COLOR_BGR2RGB)),plt.title("KeyPoint match result:")
    plt.show()

    srcPts = np.array(srcPts, dtype=np.float32).reshape(-1, 2)
    dstPts = np.array(dstPts, dtype=np.float32).reshape(-1, 2)

    return srcPts,dstPts


# 根据关键点的对应关系（srcPts，dstPts），将两张图片调整到一样的位置
def adjustImagesToSamePosWithSrcDst(image1,image2,srcPts,dstPts):
    # 计算变换矩阵
    M, mask = cv2.findHomography(dstPts, srcPts, cv2.RANSAC, 5.0)

    # 应用变换矩阵
    h, w, d = image1.shape
    image2_aligned = cv2.warpPerspective(image2, M, (w, h))

    # 找到有效区域
    Grayimg2 = cv2.cvtColor(image2_aligned, cv2.COLOR_BGR2GRAY)  # 先要转换为灰度图片
    ret, thresh = cv2.threshold(Grayimg2, 1, 255, cv2.THRESH_BINARY)  # 这里的第二个参数要调，是阈值！
    contours, _ = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    x, y, w, h = cv2.boundingRect(contours[0])

    # 裁剪图像
    image1_cropped = image1[y:y + h, x:x + w]
    image2_aligned_cropped = image2_aligned[y:y + h, x:x + w]

    # 显示结果
    plt.figure(figsize=(2*3,1*3))
    plt.title("Image Adjust Result")
    plt.subplot(121), plt.imshow(image1), plt.title("image1")
    plt.subplot(122), plt.imshow(image2_aligned_cropped), plt.title("image2_aligned_cropped")
    plt.show()
    return image1_cropped,image2_aligned_cropped

