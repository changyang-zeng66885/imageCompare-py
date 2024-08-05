from Angle import imagePosAdjust as ia
import cv2
import numpy as np
import matplotlib.pyplot as plt

image1 = cv2.imread("../../images/kuiguangta2/image1.png")
image2 = cv2.imread("../../images/kuiguangta2/image3.png")

unchangedPoints = [(136,437),(162,445),(208,438),(141,454),(225,455),(227,480),(128,487),(228,486),(94,495)]
trackPoint = [(178,38),(180,434)]

baseKeyPoints = [(148,34),(209,37),(236,453),(121,452)] # 基准比对图的关键点 顺序为：左上，右上，右下，左下
standardKeyPoints = [(188,78),(241,81),(253,626),(178,626)] # 标准距离

ROWS = 2 # 绘制子图的行数
COLS = 3 # 绘制子图的列数


#　根据不变的关键点，匹配两张图片的关键点对应关系
srcPts,dstPts,kp1,kp2 = ia.manualKeyPoints(image1,image2,unchangedPoints,50,99999)

img1_kp_unchanged = ia.showkpInImage(kp1,image1,unchangedPoints,(0,0,255))
img1_kp_tracked  = ia.showkpInImage(kp1,img1_kp_unchanged,trackPoint,(255,0,0))


#　根据不变关键点的对应关系，对拍摄图片进行位置校正
image1_cropped,image2_aligned_cropped = ia.adjustImagesToSamePosWithSrcDst(image1,image2,srcPts,dstPts)

# 对于位置校正后的图片，根据想要跟踪的关键点，识别差异
srcPts_t,dstPts_t,kp1_t,kp2_t = ia.manualKeyPoints(image1_cropped,image2_aligned_cropped,trackPoint,50,5)
img1_kp_tracked  = ia.showkpInImage(kp1_t,img1_kp_tracked,trackPoint,(255,0,0))

# print(f"srcPts_t: {srcPts_t}")
# print(f"dstPts_t: {dstPts_t}")

# # 将图上距离转化为实际距离
basePts = np.array([[148, 34], [209, 37], [236, 453], [121, 452]],dtype='float32') # 基准图上的关键点
standardPts = np.array([[188, 78], [241, 81], [253, 626], [178, 626]],dtype='float32')  # 标准图（例如图纸）上的关键点，和前面基准图上的关键点一一对应
pix2ActDistanceRate = 52.67/(625-6) # 标准图上像素距离转化为实际距离的转化比例
actualDistanceList = ia.getDistanceFromStandardImage(basePts,standardPts,srcPts_t,dstPts_t,pix2ActDistanceRate) # 根据基准图和标准图关键点的对应关系（basePts,standardPts），计算观测图相较于基准图（srcPts_t,dstPts_t）的位移距离
print("Move Distance:",actualDistanceList)

moveTrack1 = ia.showPointMatchWithArrowLine(srcPts,dstPts,image2)
moveTrack2 = ia.showPointMatchWithArrowLine(srcPts_t,dstPts_t,image2_aligned_cropped,actualDistanceList)


# 拟合直线，计算夹角
m1, c1 = ia.fit_line(srcPts_t)
# print(f"src lines:{m1}, {c1}")

m2, c2 = ia.fit_line(dstPts_t)
# print(f"dst lines:{m2}, {c2}")


ia.draw_line(moveTrack2, m2, c2, (0, 255, 0))  # 绿色直线
ia.draw_line(moveTrack2, m1, c1, (255, 0, 0))  # 红色直线

# 计算夹角
angle = np.arctan(abs((m2 - m1) / (1 + m1 * m2))) * 180 / np.pi
print(f"Angle: {angle:.4f} (Degrees)")

# 标注夹角
cv2.putText(moveTrack2, f'Angle: {angle:.2f} degrees', (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

plt.figure(figsize=(3*COLS,4*ROWS)) # 3行2列
plt.subplot(231),plt.imshow(img1_kp_tracked),plt.title("initial image1 with unchanged(blue) and track(red)")
plt.subplot(232),plt.imshow(image2),plt.title("initial image2")
plt.subplot(234),plt.imshow(image1_cropped),plt.title("image1_cropped")
plt.subplot(235),plt.imshow(image2_aligned_cropped),plt.title("image2_aligned_cropped")
plt.subplot(233),plt.imshow(moveTrack1),plt.title("moveTrack (I) (Begin with Red ,End with Green )")
plt.subplot(236),plt.imshow(moveTrack2),plt.title("moveTrack (II) (Begin with Red ,End with Green )")
plt.show()

plt.imshow(moveTrack2)
plt.show()








