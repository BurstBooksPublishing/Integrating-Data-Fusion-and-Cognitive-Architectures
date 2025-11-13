import torch, onnx, subprocess
from torch import nn
import torch.nn.utils.prune as prune
from onnxruntime.quantization import quantize_dynamic, QuantType

# Simple perception head used in a fusion node (replace with real model)
class PerceptionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.fc = nn.Linear(16*32*32, 128)
    def forward(self, x):
        x = self.conv(x).relu()
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = PerceptionHead().eval()
# Structured pruning: remove entire channels in conv layer (30%)
prune.ln_structured(model.conv, name='weight', amount=0.3, n=2, dim=0)
prune.remove(model.conv, 'weight')  # make mask permanent

# Export to ONNX (use representative input shapes)
dummy = torch.randn(1,3,32,32)
onnx_path = "perception_head.onnx"
torch.onnx.export(model, dummy, onnx_path, opset_version=13)

# Dynamic quantization using ONNX Runtime (weights quantized only)
q_onnx = "perception_head.quant.onnx"
quantize_dynamic(onnx_path, q_onnx, weight_type=QuantType.QInt8)

# Optional: autotune / build TensorRT engine if trtexec exists
try:
    subprocess.check_call([
        "trtexec",  # TensorRT CLI; available on Nvidia platforms
        "--onnx={}".format(q_onnx),
        "--saveEngine=perception_head.engine",
        "--workspace=2048", "--fp16"  # request fp16 kernels where available
    ])
except FileNotFoundError:
    print("trtexec not found; skip TensorRT autotune. Deploy quantized ONNX instead.")