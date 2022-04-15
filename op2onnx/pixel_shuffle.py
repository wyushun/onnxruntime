# Some standard imports
import numpy as np
import torch.onnx
import torch.nn as nn
import torch.nn.init as init


class OpNet(nn.Module):
    def __init__(self, upscale_factor, inplace=False):
        super(OpNet, self).__init__()
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, input):
        x = self.pixel_shuffle(input)
        return x

# Create the super-resolution model by using the above model definition.
m = OpNet(upscale_factor=3)
m.eval()

# Input to the model
batch_size = 1    # just a random number
x = torch.randn(batch_size, 9, 224, 224, requires_grad=True)
# torch_out = m(x)

# Export the model
onnx_model_name = "PixelShuffle.onnx"
torch.onnx.export(m,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  onnx_model_name,           # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=False,  # whether to execute constant folding for optimization
                  input_names=['input'],   # the model's input names
                  output_names=['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})

# do onnx model check
import onnx
onnx_model = onnx.load(onnx_model_name)
onnx.checker.check_model(onnx_model)

