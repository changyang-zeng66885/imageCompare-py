import cv2
import numpy as np
import matplotlib.pyplot as plt

def track_multiple_objects(image1_path, image2_path, targets):
    # 读取第一张图片和第二张图片
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    results = []

    for (x, y, w, h) in targets:
        # 从第一张图片中提取模板
        template = img1[y:y + h, x:x + w]

        # 使用模板匹配在第二张图片中查找模板
        result = cv2.matchTemplate(img2, template, cv2.TM_CCOEFF_NORMED)

        # 获取匹配结果中最大值的坐标
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        # 在第二张图片中画出矩形框
        cv2.rectangle(img2, top_left, bottom_right, (0, 255, 0), 2)

        # 保存结果坐标
        results.append((top_left[0], top_left[1], w, h))
        # 在第二张图片中画出矩形框
        cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(img2, top_left, bottom_right, (0, 255, 0), 2)


        cv2.arrowedLine(img2,(int(x+w/2), int(y+h/2)),(int(top_left[0]+w/2), int(top_left[1]+h/2)),(255,0,0),2,cv2.LINE_8,0,0.05)

    # 显示结果图片
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1), plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)), plt.title("Original Image")
    plt.subplot(1, 2, 2), plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)), plt.title("Track Result")
    plt.show()

    return results


# 示例使用
image1_path = '../images/bottle/image1.jpg'
image2_path = '../images/bottle/image3.jpg'
targets = [
    (593, 690, 150, 100),  # 示例目标1
    (579,1034,171,102),    # 示例目标2
    (525, 1315, 284, 127)  # 示例目标3
    # 可以添加更多目标
]

coords_list = track_multiple_objects(image1_path, image2_path, targets)
for i, coords in enumerate(coords_list):
    print(f'Object {i + 1} {targets[i]} found at: {coords}')