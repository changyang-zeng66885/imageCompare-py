from predict import getImage
from torchvision import transforms

officialMaskPath = "CrackDetect/imageSrc/masks/crack-o-287.png"
myMaskPath = "CrackDetect/bailushuyuan/labels/20210320150913.png"

#读取图片并预处理
officialMask = getImage(officialMaskPath,transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
]),"RGB") # 将图片转换为模型输入所需的格式

myMask = getImage(myMaskPath,transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
]),"RGB") # 将图片转换为模型输入所需的格式

print(f"officialMask:{officialMask}")
print(f"myMask:{myMask}")