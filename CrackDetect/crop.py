import os
import json
import cv2
import numpy as np

# 设置输入输出目录
input_dir = 'bailushuyuan/images'
output_dir = 'bailushuyuan/labels'
label_dir = "bailushuyuan/json"

# 遍历所有 .json 文件
for filename in os.listdir(label_dir):
    if filename.endswith('.json'):
        # 读取 .json 文件
        print("filename:", filename)
        try:
            with open(os.path.join(label_dir, filename), 'r',encoding='utf-8') as f:
                data = json.load(f)
        except:
            raise Exception(f"can not open {filename}")


        # 获取对应的图像文件名
        image_path = input_dir +"/"+os.path.splitext(filename)[0] + '.jpg'
        print(f"image_path:{image_path}")

        # 读取图像
        image = cv2.imread(image_path)
        # 创建一个与原图像大小相同的背景板
        canvas = np.zeros_like(image, dtype=np.uint8)
        canvas[:] = 0  # 将所有像素值设置为0

        # 遍历所有的 "Crack" 标签
        for shape in data['shapes']:
            if shape['label'] == 'Crack':
                # 获取多边形顶点坐标
                points = np.array([shape['points']], dtype=np.int32)
                print(f"shape['points']:{shape['points']}")
                print(f"canvas.shape:{canvas.shape}")

                # 在画布上绘制多边形
                cv2.fillPoly(canvas, points, (127, 127, 127))

        # 将画布与原图像进行叠加

        # 保存结果图像
        output_path = output_dir +"/"+os.path.splitext(filename)[0] + '.png'
        cv2.imwrite(output_path, cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY))
        print(f'Saved {output_path}')