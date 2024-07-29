import cv2
import numpy as np
import matplotlib.pyplot as plt

import edge

image1 = cv2.imread("../images/kuiguangta2/image1.png")
image2 = cv2.imread("../images/kuiguangta2/image3.png")

img1WithEdge = edge.getEdges(image1)
img2WithEdge = edge.getEdges(image2)

plt.figure(figsize=(12,4))
plt.subplot(1,3,1),plt.imshow(img1WithEdge,cmap="gray"),plt.title("img1WithEdge")
plt.subplot(1,3,2),plt.imshow(img2WithEdge,cmap="gray"),plt.title("img2WithEdge")

# 检测白线    这里是设置检测直线的条件，可以去读一读HoughLinesP()函数，然后根据自己的要求设置检测条件
lines = cv2.HoughLinesP(img1WithEdge, 1, np.pi / 180, 40, minLineLength=10, maxLineGap=10)
print("lines=", lines)
print("========================================================")
i = 1
# 对通过霍夫变换得到的数据进行遍历
for line in lines:
    # newlines1 = lines[:, 0, :]
    print("line[" + str(i - 1) + "]=", line)
    x1, y1, x2, y2 = line[0]  # 两点确定一条直线，这里就是通过遍历得到的两个点的数据 （x1,y1）(x2,y2)
    cv2.line(image1, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 在原图上画线
    # 转换为浮点数，计算斜率
    x1 = float(x1)
    x2 = float(x2)
    y1 = float(y1)
    y2 = float(y2)
    print("x1=%s,x2=%s,y1=%s,y2=%s" % (x1, x2, y1, y2))
    if x2 - x1 == 0:
        print("直线是竖直的")
        result = 90
    elif y2 - y1 == 0:
        print("直线是水平的")
        result = 0
    else:
        # 计算斜率
        k = -(y2 - y1) / (x2 - x1)
        # 求反正切，再将得到的弧度转换为度
        result = np.arctan(k) * 57.29577
        print("直线倾斜角度为：" + str(result) + "度")
    i = i + 1
# 显示最后的成果图
plt.subplot(1,3,3),plt.imshow(image1,cmap="gray"),plt.title("img2WithEdge")
plt.show()
