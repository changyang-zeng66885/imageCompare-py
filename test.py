import cv2
import numpy as np
import matplotlib.pyplot as plt
from  autoKeyPoints import moveDetectionByKeyPoints
from manualKeyPoints import manualKeyPoints

# 读取两张照片
image1 = cv2.imread('images/kuiguangta/image1.jpg')
image2 = cv2.imread('images/kuiguangta/image3.jpg')
# 在照片1中手动指定关键点坐标
specified_keypoints_coords = [(713,1153),(717,1221),(1009,1085),(1057,985),(1161,981),(1325,1085),(1529,1113),(1605,1213),
                              (1129,117),(789,277)]


srcPts,dstPts = manualKeyPoints(image1,image2,specified_keypoints_coords)

# print("srcpts",srcpts)
# print("dstpts",dstpts)

# 计算变换矩阵
M, mask = cv2.findHomography(dstPts, srcPts, cv2.RANSAC, 5.0)

# 应用变换矩阵
h, w, d = image1.shape
image2_aligned = cv2.warpPerspective(image2, M, (w, h))

# 找到有效区域
Grayimg2 = cv2.cvtColor(image2_aligned, cv2.COLOR_BGR2GRAY) # 先要转换为灰度图片
ret, thresh = cv2.threshold(Grayimg2, 1, 255,cv2.THRESH_BINARY) # 这里的第二个参数要调，是阈值！
contours, _ = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

x, y, w, h = cv2.boundingRect(contours[0])

# 裁剪图像
image1_cropped = image1[y:y+h, x:x+w]
image2_aligned_cropped = image2_aligned[y:y+h, x:x+w]

# 显示结果
plt.subplot(121),plt.imshow(image1),plt.title("image1")
plt.subplot(122),plt.imshow(image2_aligned_cropped),plt.title("image2_aligned_cropped")
plt.show()

moveDetectionByKeyPoints(image1_cropped,image2_aligned_cropped)