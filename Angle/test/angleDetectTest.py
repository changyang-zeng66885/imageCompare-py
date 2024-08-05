from Angle import imagePosAdjust as ia
import cv2
import numpy as np
import matplotlib.pyplot as plt

image1 = cv2.imread("../../images/shanzi/image1.jpg")
image2 = cv2.imread("../../images/shanzi/image3.jpg")

unchangedPoints = [(587,1491),(629,1487),(671,1485),(697,1413),(573,1393),(449,1545),(799,1531),(183,1535),(1017,1545),(1081,1441),(87,1457)]
trackPoint = [(685,97),(685,301),(663,549),(651,777),(614,1127)]


ROWS = 2
COLS = 3
plt.figure(figsize=(3*COLS,4*ROWS)) # 3行2列

#　根据不变的关键点，匹配两张图片的关键点对应关系
srcPts,dstPts,kp1,kp2 = ia.manualKeyPoints(image1,image2,unchangedPoints,100,99999)

img1_kp_unchanged = ia.showkpInImage(kp1,image1,unchangedPoints,(0,0,255))
img1_kp_tracked  = ia.showkpInImage(kp1,img1_kp_unchanged,trackPoint,(255,0,0))
plt.subplot(231),plt.imshow(img1_kp_tracked),plt.title("initial image1 with unchanged(blue) and track(red)")
plt.subplot(232),plt.imshow(image2),plt.title("initial image2")

#　根据不变关键点的对应关系，对拍摄图片进行位置校正
image1_cropped,image2_aligned_cropped = ia.adjustImagesToSamePosWithSrcDst(image1,image2,srcPts,dstPts)


plt.subplot(234),plt.imshow(image1_cropped),plt.title("image1_cropped")
plt.subplot(235),plt.imshow(image2_aligned_cropped),plt.title("image2_aligned_cropped")

# 对于位置校正后的图片，根据想要跟踪的关键点，识别差异
srcPts_t,dstPts_t,kp1_t,kp2_t = ia.manualKeyPoints(image1_cropped,image2_aligned_cropped,trackPoint,50,10)

print(f"srcPts_t: {srcPts_t}")
print(f"dstPts_t: {dstPts_t}")

moveTrack1 = ia.showPointMatchWithArrowLine(srcPts,dstPts,image2)
moveTrack2 = ia.showPointMatchWithArrowLine(srcPts_t,dstPts_t,image2_aligned_cropped)

# 拟合直线，计算夹角
m1, c1 = ia.fit_line(srcPts_t)
# print(f"src lines:{m1}, {c1}")

m2, c2 = ia.fit_line(dstPts_t)
# print(f"dst lines:{m2}, {c2}")


ia.draw_line(moveTrack2, m2, c2, (0, 255, 0))  # 绿色直线
ia.draw_line(moveTrack2, m1, c1, (255, 0, 0))  # 红色直线

# 计算夹角
angle = np.arctan(abs((m2 - m1) / (1 + m1 * m2))) * 180 / np.pi
print(f"Angle: {angle}")

# 标注夹角
cv2.putText(moveTrack2, f'Angle: {angle:.2f} degrees', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 5, cv2.LINE_AA)

plt.subplot(233),plt.imshow(moveTrack1),plt.title("moveTrack (I) (Begin with Red ,End with Green )")
plt.subplot(236),plt.imshow(moveTrack2),plt.title("moveTrack (II) (Begin with Red ,End with Green )")
plt.show()








