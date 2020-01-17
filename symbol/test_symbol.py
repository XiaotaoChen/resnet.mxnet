import mxnet as mx

def test_symbol(num_classes=10):
    input_data = mx.sym.Variable(name="data")
    flatten = mx.sym.Flatten(data=input_data)
    fc1 = mx.sym.FullyConnected(name='fc1', data=flatten, num_hidden=32, no_bias=1)
    fc2 = mx.sym.FullyConnected(name='fc2', data=fc1, num_hidden=num_classes, no_bias=1)
    softmax = mx.sym.SoftmaxOutput(data=fc2, name='softmax')
    return softmax