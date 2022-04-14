import torch
from torchvision.models import alexnet
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models import inception_v3
from thop import profile
input = torch.randn(1, 3, 224, 224)

# models = [resnet18(), resnet34(), resnet50(), resnet101(), resnet152()]
models = [inception_v3()]

for model in models:
    print("--------------------------------------------------------------------")
    flops, params = profile(model, inputs=(input, ))
    print('FLOPs = ' + str(flops/1000**3) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')
