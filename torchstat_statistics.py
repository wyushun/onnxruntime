from torchstat import stat
from torchvision.models import alexnet
from torchvision.models import vgg16, vgg19
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models import inception_v3, googlenet
from torchvision.models import mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large

# models = [vgg16(), vgg19(), resnet18()]
# models = [resnet34(), resnet50(), resnet101(), resnet152()]
# models = [alexnet()]
# models = [inception_v3(), googlenet()]
models = [mobilenet_v2(), mobilenet_v3_small(), mobilenet_v3_large()]
input_shape = (3, 224, 224)
for model in models:
    print("--------------------------------------------------------------------")
    stat(model, input_shape)
