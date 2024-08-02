import cv2
import numpy as np
import os

"""
    使用网络摄像头，采集图像
    按's'保存图片
    按'q'退出图像采集
"""

# 初始化两个摄像头
cam1 = cv2.VideoCapture(0)
cam2 = cv2.VideoCapture(1)

# 保存图像的文件夹
save_folder = 'images/demoImage6/test/'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# 初始化图像计数器
img_count = 0
print("Enter `s` to shoot; Enter`q` to quit")
while True:
    # 读取两个摄像头的画面
    ret1, frame1 = cam1.read()
    ret2, frame2 = cam2.read()

    # 显示两个摄像头的画面
    cv2.imshow('Camera 1', frame1)
    cv2.imshow('Camera 2', frame2)


    # 按下's'键保存图像
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        # 保存左摄像头的图像
        cv2.imwrite(os.path.join(save_folder, f'left/left_{img_count}.jpg'), frame1)
        # 保存右摄像头的图像
        cv2.imwrite(os.path.join(save_folder, f'right/right_{img_count}.jpg'), frame2)
        print(f'Saved images {img_count}')
        img_count += 1

    if key == ord('q'):
        break


# 释放摄像头资源
cam1.release()
cam2.release()
cv2.destroyAllWindows()