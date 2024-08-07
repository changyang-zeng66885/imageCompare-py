# 收集所有图像路径和标签
import os

root_dir = 'CloudSeaScene/images'
for label, class_name in enumerate(os.listdir(root_dir)):
    print(f"label:{label},class_name:{class_name}")

"""
label:0,class_name:cloudsea
label:1,class_name:scene
label:2,class_name:foguang
label:3,class_name:sunrise
"""