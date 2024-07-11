import cv2
import numpy as np
import matplotlib.pyplot as plt
def moveDetectionByKeyPoints(image1,image2):
    # 读取图片
    # 将图片转换为灰度图
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # 初始化SIFT检测器
    sift = cv2.SIFT_create()

    # 检测关键点和计算描述子
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # 使用FLANN匹配器进行匹配
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # 使用Lowe's ratio test来筛选匹配点
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # 提取匹配点坐标
    points1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    points2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 计算单应性矩阵
    H, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    # 计算关键点位置的变化
    point_changes = []
    for i, match in enumerate(good_matches):
        if matchesMask[i]:
            pt1 = np.array(kp1[match.queryIdx].pt)
            pt2 = np.array(kp2[match.trainIdx].pt)
            point_changes.append((pt1, pt2, np.linalg.norm(pt1 - pt2)))

    # 定义绘制参数
    keypoint_size = 10
    line_thickness = 5

    # 在图片上绘制关键点和匹配线
    image1_with_kp = cv2.drawKeypoints(image1, kp1, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    image2_with_kp = cv2.drawKeypoints(image2, kp2, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # 绘制匹配线条
    for i, match in enumerate(good_matches):
        if matchesMask[i]:
            pt1 = tuple(np.round(kp1[match.queryIdx].pt).astype(int))
            pt2 = tuple(np.round(kp2[match.trainIdx].pt).astype(int) + np.array([image1.shape[1], 0]))
            cv2.circle(image1_with_kp, pt1, keypoint_size, (0, 0, 255), -1)
            cv2.circle(image2_with_kp, pt2 - np.array([image1.shape[1], 0]), keypoint_size, (0, 0, 255), -1)
            cv2.line(image2_with_kp, pt1, pt2 - np.array([image1.shape[1], 0]), (255, 0, 0), line_thickness)

    # 合并两张图片用于显示匹配
    combined_image = np.hstack((image1_with_kp, image2_with_kp))

    # 显示结果
    plt.figure(figsize=(20, 10))
    plt.title('Auto Keypoints and Matches')
    plt.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
    plt.show()



