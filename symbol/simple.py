import mxnet as mx 

def conv_bn_relu(data, name, stride=(1,1)):
    sym = mx.sym.Convolution(data=data, num_filter=8, kernel=(3, 3), stride=stride, pad=(1, 1),
                              no_bias=True, name=name + "_conv")
    bn1 = mx.sym.BatchNorm(data=sym, fix_gamma=False, name=name + '_bn')
    relu1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu')
    return relu1

def cifar10_sym():
    data = mx.sym.Variable(name='data')
    sym = conv_bn_relu(data, name = "stage1", stride=(2,2))
    sym = conv_bn_relu(sym, name = "stage2", stride=(2,2))
    pool1 = mx.symbol.Pooling(data=sym, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')
    flat = mx.symbol.Flatten(data=pool1)
    fc1 = mx.symbol.FullyConnected(data=flat, num_hidden=10, name='fc1')
    cls = mx.symbol.SoftmaxOutput(data=fc1, name='softmax')
    return cls