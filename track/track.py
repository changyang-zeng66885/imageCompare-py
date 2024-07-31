import cv2
import numpy as np
import matplotlib.pyplot as plt

def track_object(image1_path, image2_path, x, y, w, h):
    # 读取第一张图片和第二张图片
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    # 从第一张图片中提取模板
    template = img1[y:y+h, x:x+w]

    # 使用模板匹配在第二张图片中查找模板
    result = cv2.matchTemplate(img2, template, cv2.TM_CCOEFF_NORMED)

    # 获取匹配结果中最大值的坐标
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # 在第二张图片中画出矩形框
    cv2.rectangle(img1, (x,y), (x+w,y+h), (0, 255, 0), 2)
    cv2.rectangle(img2, top_left, bottom_right, (0, 255, 0), 2)

    # 显示结果图片
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1),plt.imshow(cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)),plt.title("Original Image")
    plt.subplot(1,2,2),plt.imshow(cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)),plt.title("Track Result")
    plt.show()

    # 返回矩形框的坐标
    return top_left[0], top_left[1], w, h

# 示例使用
image1_path = '../images/bottle/image1.jpg'
image2_path = '../images/bottle/image4.jpg'
x, y, w, h = 593, 690, 150, 100  # 示例坐标，请根据实际情况调整

coords = track_object(image1_path, image2_path, x, y, w, h)
print(f'Object found at: {coords}')