import torch
from model import CNNClasification # 导入与训练时相同的模型类


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:",device)
model = CNNClasification().to(device)
model.load_state_dict(torch.load('CloudSeaScene/model_saved/CnnClasification_20240807_epoch5_acc09057.pth'))

model.eval()
dummy_input = torch.randn(1, 3, 128, 128).to(device)


torch.onnx.export(model, dummy_input, "CloudSeaScene/model_saved/CnnClasification_20240807_epoch5_acc09057.onix")