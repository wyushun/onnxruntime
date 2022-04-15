import onnx
import numpy as np
import torch.onnx
import torch.nn as nn
import torch.nn.init as init


m = nn.Transformer(nhead=16, num_encoder_layers=12)
m.eval()
src = torch.rand((10, 32, 512))
tgt = torch.rand((20, 32, 512))
out = m(src, tgt)

# Export the model
onnx_model_name = "Transformer.onnx"
torch.onnx.export(m,               # model being run
                  (src, tgt),                         # model input (or a tuple for multiple inputs)
                  onnx_model_name,           # where to save the model (can be a file or file-like object)
                  export_params=False,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=False,  # whether to execute constant folding for optimization
                  input_names = ['src', 'tgt'],   # the model's input names
                  output_names = ['output']) # the model's output names


onnx_model = onnx.load(onnx_model_name)
onnx.checker.check_model(onnx_model)
