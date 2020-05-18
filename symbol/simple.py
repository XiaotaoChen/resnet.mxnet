import mxnet as mx
import numpy as np


def get_constant(value):
    init_str = '[\"constant\", {\"value\": ' + str(value) + '}]'
    return init_str

def conv_bn_relu(data, name, stride=(1,1)):
    sym = mx.sym.Convolution(data=data, num_filter=8, kernel=(3, 3), stride=stride, pad=(1, 1),
                              no_bias=True, name=name + "_conv")
    bn1 = mx.sym.BatchNorm(data=sym, fix_gamma=False, name=name + '_bn')
    relu1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu')
    return relu1

def cifar10_sym():
    data = mx.sym.Variable(name='data')
    flat = mx.symbol.Flatten(data=data)
    # sym = conv_bn_relu(data, name = "stage1", stride=(2,2))
    # sym = conv_bn_relu(sym, name = "stage2", stride=(2,2))
    # pool1 = mx.symbol.Pooling(data=sym, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')
    # flat = mx.symbol.Flatten(data=pool1)
    fc1 = mx.symbol.FullyConnected(data=flat, num_hidden=10, name='fc1')
    cls = mx.symbol.SoftmaxOutput(data=fc1, name='softmax')
    return cls

def do_kurt(var, shape, kT=1.8, num_layers=1, lambd=1.0):
    cnt = 1
    for dim in shape:
        cnt *= dim
    mean = mx.sym.mean(name=var.name+"_mean", data=var)
    meaned_var = mx.sym.broadcast_sub(name=var.name+"_broadcast_sub", lhs=var, rhs=mean)
    norm_var = mx.sym.norm(name=var.name+"_norm", data=meaned_var)

    # meaned_std = 1.0/np.sqrt(cnt) * norm_var
    # kurt = 1.0/cnt * mx.sym.sum((mx.sym.broadcast_div(meaned_var, meaned_std))**4)
    # kurt_loss = lambd/num_layers * (kurt - kT)
    # return mx.sym.make_loss(kurt_loss, grad_scale=1)

    kurt_val = cnt * mx.sym.sum(name=var.name+"_sum", data=(mx.sym.broadcast_div(meaned_var, norm_var)) **4)
    kurt_loss = (kurt_val - kT)**2

    return mx.sym.make_loss(name=var.name+"_loss", data=kurt_loss, grad_scale= lambd / num_layers)

def do_kurt_np(data):
    total_count  = 1
    for i in data.shape:
        total_count *= i
    mean = np.mean(data)
    std = np.std(data)
    kurt_val = np.sum(((data - mean)/std)**4) / total_count
    kurt_loss = (kurt_val - 1.8)**2
    return kurt_loss

def do_kurt_np_v2(data):
    total_count  = 1
    for i in data.shape:
        total_count *= i
    # std = sqrt(1/n * sum((x-u)**2)), norm = sqrt(sum(x**2))
    mean = np.mean(data)
    meaned_data = data - mean
    norm_data = np.linalg.norm(meaned_data)
    kurt_val = total_count * np.sum((meaned_data/norm_data)**4)
    kurt_loss = (kurt_val - 1.8)**2
    return kurt_loss





def cifar10_sym_kurt():
    data = mx.sym.Variable(name='data')
    flat = mx.symbol.Flatten(data=data)
    # sym = conv_bn_relu(data, name = "stage1", stride=(2,2))
    # sym = conv_bn_relu(sym, name = "stage2", stride=(2,2))
    # pool1 = mx.symbol.Pooling(data=sym, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')
    # flat = mx.symbol.Flatten(data=pool1)
    shape = (10,3072)
    fc_weight = mx.symbol.var(name="fc1_weight", shape=shape)
    kurt_loss = do_kurt(fc_weight, shape=shape, lambd=1.0)

    fc1 = mx.symbol.FullyConnected(data=flat, weight=fc_weight, num_hidden=10, name='fc1')
    cls = mx.symbol.SoftmaxOutput(data=fc1, name='softmax')

    outputs = [kurt_loss, cls]
    outputs = mx.sym.Group(outputs)
    return outputs