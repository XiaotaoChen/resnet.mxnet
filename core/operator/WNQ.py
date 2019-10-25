import mxnet as mx
import numpy as np
import copy

def simulate_wnq(data, is_perchannel, QUANT_LEVEL):
    if is_perchannel:
        reduce_axis = tuple([i for i in range(len(data.shape))])
        max_abs = np.max(np.abs(data), axis=reduce_axis[1:])
        target_shape = (data.shape[0],) + (1,) * len(data.shape[1:])
        max_abs =max_abs.reshape(target_shape)
    else:
        max_abs = np.max(np.abs(data))
    normed_data = data/max_abs
    quanted_data = np.round(normed_data * QUANT_LEVEL) / QUANT_LEVEL * max_abs
    return quanted_data

def simulate_wnq_backword(data, out_grad, is_perchannel, QUANT_LEVEL):
    data_abs = np.abs(data)
    # print("data:\n{}\nout grad:\n{}".format(data, out_grad))

    if is_perchannel:
        reduce_axis = tuple([i for i in range(len(data.shape))])
        max_abs = np.max(np.abs(data), axis=reduce_axis[1:])
        target_shape = (data.shape[0],) + (1,) * len(data.shape[1:])
        max_abs =max_abs.reshape(target_shape)
        max_abs_flag = data_abs == max_abs
        exp_max_flag = data_abs != max_abs
        max_abs_grad = - np.sum(out_grad * data * exp_max_flag, axis=reduce_axis[1:]).reshape(target_shape) / max_abs
        # max_abs_grad = max_abs_grad.reshape(target_shape)
        data_grad = out_grad * exp_max_flag + max_abs_flag * max_abs_grad
    else:
        max_abs = np.max(data_abs)
        max_abs_flag = data_abs == max_abs
        exp_max_flag = data_abs != max_abs
        max_abs_grad = - np.sum(out_grad * data * exp_max_flag) / max_abs
        data_grad = out_grad * exp_max_flag + max_abs_flag * max_abs_grad

    return data_grad


def print_info(auto_grad, cal_grad, name):
    print("{} autograd:\n{}\n cal grad:\n{}".format(name, auto_grad, cal_grad))
    print("{} autograd - cal_grad:\n{}".format(name, auto_grad - cal_grad))


class WNQ_PY(mx.operator.CustomOp):
    def __init__(self, nbits, is_perchannel):
        self.nbits = nbits
        self.is_perchannel = is_perchannel
        self.QUANT_LEVEL = 2**self.nbits - 1
    def forward(self, is_train, req, in_data, out_data, aux):
        data = in_data[0]
        data_abs = mx.nd.abs(data)
        if self.is_perchannel is False:
            max_abs = mx.nd.max(data_abs)
        else:
            reduce_axis = tuple([i for i in range(len(data.shape))])
            max_abs = mx.nd.max(data_abs, axis=reduce_axis[1:])
            target_shape = (data.shape[0],) + (1,) * len(data.shape[1:])
            max_abs = max_abs.reshape(target_shape)

        normed_data = data / max_abs
        self.assign(out_data[0], req[0], mx.nd.round(normed_data * self.QUANT_LEVEL) / self.QUANT_LEVEL * max_abs)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        data = in_data[0]
        data_abs = mx.nd.abs(data)
        
        if self.is_perchannel is False:
            max_abs_data = mx.nd.max(data_abs)
            exp_max_abs_flag = data_abs != max_abs_data
            max_abs_flag = data_abs == max_abs_data
            max_abs_grad = - mx.nd.sum(out_grad[0] * data * exp_max_abs_flag) / max_abs_data
        else:
            reduce_axis = tuple([i for i in range(len(data.shape))])
            max_abs_data = mx.nd.max(data_abs, axis=reduce_axis[1:])
            target_shape = (data.shape[0],) + (1,) * len(data.shape[1:])
            max_abs_data = max_abs_data.reshape(target_shape)
            
            exp_max_abs_flag = data_abs != max_abs_data
            max_abs_flag = data_abs == max_abs_data
            
            max_abs_grad = - mx.nd.sum(out_grad[0] * data * exp_max_abs_flag, axis=reduce_axis[1:]).reshape(target_shape) / max_abs_data

        self.assign(in_grad[0], req[0], out_grad[0] * exp_max_abs_flag + max_abs_grad * max_abs_flag)

        # simulated_grad = simulate_wnq_backword(data.asnumpy(), out_grad[0].asnumpy(), self.is_perchannel, self.QUANT_LEVEL)
        # print_info(in_grad[0].asnumpy(), simulated_grad, "grad")


@mx.operator.register("WNQ_PY")
class WNQ_PYProp(mx.operator.CustomOpProp):
    def __init__(self, nbits=4, is_perchannel=False):
        self.nbits = int(nbits)
        self.is_perchannel = eval(is_perchannel)
        super(WNQ_PYProp, self).__init__(True)
    def list_arguments(self):
        return ['data']
    def list_outputs(self):
        return ['output']
    def list_auxiliary_states(self):
        return []
    def infer_shape(self, in_shape):
        shape = in_shape[0]
        return [shape], [shape], []
    def infer_type(self, in_type):
        dtype = in_type[0]
        return [dtype] * len(in_type), [in_type[0]]*len(self.list_outputs()), \
            [in_type[0]]*len(self.list_auxiliary_states())

    def create_operator(self, ctx, shapes, dtypes):
        return WNQ_PY(self.nbits, self.is_perchannel)
                


