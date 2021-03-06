from .resnet import resnet
from .resnet import resnet_cifar10
from .resnet import preact_resnet
from .resnext import resnext
from .resnext_cyt import resnext as resnext_cyt
from .resnet_mxnet import resnet_mxnet
from .resnet_int8 import resnet_int8
from .mobilenet import mobilenet
# from .mobilenet_int8 import mobilenet_int8
# from .mobilenet_int8_foldbn import mobilenet_int8_foldbn
# from .mobilenet_int8_foldbn_v1 import mobilenet_int8_foldbn_v1
# from .mobilenet_int8_clipgrad import mobilenet_int8_clipgrad
# from .mobilenet_int8_gdrq import mobilenet_int8_gdrq
from .mobilenet_int8_cxx import mobilenet_int8_cxx
from .simple import cifar10_sym