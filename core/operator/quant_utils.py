import mxnet as mx
import numpy as np
import copy


def print_info(auto_grad, cal_grad, name):
    print("{} autograd:\n{}\n cal grad:\n{}".format(name, auto_grad.asnumpy(), cal_grad.asnumpy()))
    print("{} autograd - cal_grad:\n{}".format(name, auto_grad.asnumpy() - cal_grad.asnumpy()))


def quantizeK(data, nbits):
    QUANT_LEVEL = 2**nbits - 1
    return mx.nd.round(QUANT_LEVEL * data) / QUANT_LEVEL
