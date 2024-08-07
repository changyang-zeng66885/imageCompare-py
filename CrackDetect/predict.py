import torch
from model import UNet # 导入与训练时相同的模型类
from torchvision import transforms
from PIL import Image

def getImage(imagePath,transform,colorType="RGB"):
    # colorType="RGB"，"L""
    image = Image.open(imagePath).convert(colorType)

    if transform:
        image = transform(image)
    return image

def tensor4DToImage(tensorImage,savePath):
    image = tensorImage.squeeze(0).squeeze(0).cpu()
    # 将 tensor 转换为 PIL 图像
    pil_image = Image.fromarray(image.byte().numpy(), mode='L')

    # 保存为 PNG 文件
    pil_image.save(savePath)
    print(f"Result saved at `{savePath}`")

imagePath = "imageCompare-py/images/bailushuyuan/20240412103152_cropped.jpg"
savePath = "imageCompare-py/CrackDetect/imageSrc/predictResults/result_20240412103152_cropped.jpg"


# 1. 创建模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:",device)
model = UNet(in_channels=3, out_channels=1).to(device)

# 2. 加载保存的模型参数
model.load_state_dict(torch.load('imageCompare-py/CrackDetect/model_saved/best_model_epoch5.pth'))

# 3. 设置模型为评估模式
model.eval()

# 4. 读取图片并预处理
image = getImage(imagePath,transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
]),"RGB") # 将图片转换为模型输入所需的格式

# 5. 进行预测
with torch.no_grad():
    image = image.to(device)
    images = image.unsqueeze(0)  # 现在 image 的形状是 (1, 1, 256, 256)(n,dim,rows,cols)
    outputs = model(images)
    print(outputs.size())
    tensor4DToImage(outputs,savePath)