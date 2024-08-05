from torchvision.io.image import read_image
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.transforms.functional import to_pil_image

img = read_image("../images/kuiguangta2/image1.png")
# 设置模型类型
weights = FCN_ResNet50_Weights.DEFAULT
model = fcn_resnet50(weights=weights)
model.eval()

# 初始化图片预处理函数
preprocess = weights.transforms()

# 图像预处理
batch = preprocess(img).unsqueeze(0)

# 使用模型做预测
prediction = model(batch)["out"]
normalized_masks = prediction.softmax(dim=1)
class_to_idx = {cls: idx for (idx, cls) in enumerate(weights.meta["categories"])}
mask = normalized_masks[0, class_to_idx["dog"]]
to_pil_image(img).show()
to_pil_image(mask).show()