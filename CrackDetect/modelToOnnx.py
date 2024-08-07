import torch
from model import UNet # 导入与训练时相同的模型类


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:",device)
model = UNet(in_channels=3, out_channels=1).to(device)
model.load_state_dict(torch.load('CrackDetect/model_saved/best_model_epoch_10.pth'))
model.eval()
dummy_input = torch.randn(1, 3, 256, 256).to(device)


torch.onnx.export(model, dummy_input, "CrackDetect/model_saved/best_model_epoch_10.onnx")