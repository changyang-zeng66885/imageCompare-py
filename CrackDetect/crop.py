from PIL import Image

# 打开图片
imagePath = "imageCompare-py/images/bailushuyuan/20240412103152.jpg"

image = Image.open(imagePath).convert("RGB")

# 定义裁剪范围
x, y, w, h = 184, 3, 400, 400

# 裁剪图像
cropped_image = image.crop((x, y, x+w, y+h))

# 保存裁剪后的图像
cropped_image.save("imageCompare-py/images/bailushuyuan/20240412103152_cropped.jpg")