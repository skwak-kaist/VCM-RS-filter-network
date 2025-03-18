import numpy as np
import torch
import torch.nn as nn


def init_weights(m):
    if isinstance(m, nn.Linear):
        m.weight.data = torch.Tensor(
            np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [-1.0, -1.0, 0.0, 0.0],
                    [1.0, -1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            )
        )


model = torch.nn.Sequential(nn.Linear(4, 4, bias=False))
model.apply(init_weights)
model.eval()
input0 = np.array(
    [[1.0, 2.0, 1.0, 3.0]], dtype=np.float32
)  # in sadl, tensor are nhwc...
inputs_torch = torch.from_numpy(input0).detach()
inputs_torch.requires_grad = True
output = model(inputs_torch)
print("Output", output)
print(model)
torch.onnx.export(
    model, inputs_torch, "./pytorch_matmult.onnx", verbose=True, opset_version=10
)
