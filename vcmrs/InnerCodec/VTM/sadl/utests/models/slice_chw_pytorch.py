import os
import sys
import torch


class SliceModule(torch.nn.Module):
    def __init__(self):
        super(SliceModule, self).__init__()

    def forward(self, x):
        # Slice along c, h, w
        # No slicing across n (PyTorch data format NCHW)
        y = x[:, :6, 4:70, 4:68]
        return y


input_size = (1, 10, 72, 72)
output_size = (1, 6, 66, 64)

model = SliceModule()
input = torch.randn(*input_size)
model.eval()
output = model(input)

# Output onnx model path
onnx_model_path = os.path.join(os.path.dirname(sys.argv[0]), "slice_chw_pytorch.onnx")
# Convert to onnx
torch.onnx.export(model, input, onnx_model_path, opset_version=10)
