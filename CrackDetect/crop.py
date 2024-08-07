import os
import json
import cv2
import numpy as np

# 设置输入输出目录
input_dir = 'C:/Users/minw/Desktop/train_data_crack/images'
output_dir = 'C:/Users/minw/Desktop/train_data_crack/labeled_images'
label_dir = "C:/Users/minw/Desktop/train_data_crack/labels"

# 遍历所有 .json 文件
for filename in os.listdir(label_dir):
    if filename.endswith('.json'):
        # 读取 .json 文件
        with open(os.path.join(label_dir, filename), 'r') as f:
            data = json.load(f)
            print("filename:",filename)

        # 获取对应的图像文件名
        image_path = input_dir +"/"+os.path.splitext(filename)[0] + '.jpg'
        print(f"image_path:{image_path}")

        # 读取图像
        image = cv2.imread(image_path)
        # 创建一个与原图像大小相同的全白色图像
        canvas = np.zeros_like(image, dtype=np.uint8)
        canvas[:] = 255  # 将所有像素值设置为 255 (白色)

        # 遍历所有的 "Crack" 标签
        for shape in data['shapes']:
            if shape['label'] == 'Crack':
                # 获取多边形顶点坐标
                points = np.array([shape['points']], dtype=np.int32)
                print(f"shape['points']:{shape['points']}")
                print(f"canvas.shape:{canvas.shape}")

                # 在画布上绘制多边形
                cv2.fillPoly(canvas, points, (0, 0, 0))

        # 将画布与原图像进行叠加

        # 保存结果图像
        output_path = output_dir +"/"+os.path.splitext(filename)[0] + '.jpg'
        cv2.imwrite(output_path, cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY))
        print(f'Saved {output_path}')