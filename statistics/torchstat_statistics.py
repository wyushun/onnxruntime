from torchstat import stat
from torchvision.models import alexnet
from torchvision.models import vgg16, vgg19
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models import inception_v3, googlenet
from torchvision.models import mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large
from torchvision.models import densenet121, densenet161, densenet169, densenet201
from torchvision.models import squeezenet1_0, squeezenet1_1
from torchvision.models import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7
from torchvision.models import shufflenet_v2_x1_0, shufflenet_v2_x0_5

# models = [vgg16(), vgg19(), resnet18()]
# models = [resnet34(), resnet50(), resnet101(), resnet152()]
# models = [alexnet()]
models = [shufflenet_v2_x1_0()]
# models = [mobilenet_v2(), mobilenet_v3_small(), mobilenet_v3_large()]
input_shape = (3, 224, 224)
for model in models:
    print("--------------------------------------------------------------------")
    stat(model, input_shape)
