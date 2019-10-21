import mxnet as mx
import numpy as np
import copy


def simulate_QIL_BW(data, pruning_point, clipping_point, out_grad):
    print("clip point:{}, pruning point:{}".format(clipping_point, pruning_point))
    data_abs = np.abs(data)
    data_sign = np.sign(data)
    interval_out_grad = (data_abs > pruning_point) * (data_abs < clipping_point) * out_grad
    data_grad = interval_out_grad / (clipping_point - pruning_point)
    pruning_point_grad = np.sum( interval_out_grad * ( (data - clipping_point * data_sign) / ((clipping_point - pruning_point)**2) ) )
    clipping_point_grad = np.sum(interval_out_grad * ( - (data - pruning_point * data_sign) / ((clipping_point - pruning_point)**2) ) )

def assert_all(pruning_point, clipping_point):
    pruning = pruning_point.asnumpy()
    clipping = clipping_point.asnumpy()
    # if np.all(pruning >=0) is False:
    #     pruning_point = mx.nd.zeros_like(pruning_point)
    assert np.all(pruning >= 0), "pruning {} must greater than 0".format(pruning[0])
    assert np.all(pruning < clipping), "pruning vs clipping {} vs {} pruning must less \
        than clipping".format(pruning[0], clipping[0])
    assert np.all(clipping <= 1), "clipping {} must less than 1.0".format(clipping[0])

def interval_quantize(interval_data, sign_data, pruning_point, clipping_point, quant_level):
    interval = (clipping_point - pruning_point) / quant_level
    return mx.nd.round( (interval_data - pruning_point * sign_data) / interval ) * interval + pruning_point * sign_data

class QIL_PY(mx.operator.CustomOp):
    def __init__(self, is_weight, fix_gamma, nbits):
        self.is_weight = is_weight
        self.fix_gamma = fix_gamma
        self.nbits = nbits
        self.QUANT_LEVEL = 2**self.nbits -1
        self.init = True
        self.count = 0

        self.max = 0

    def forward(self, is_train, req, in_data, out_data, aux):
        # self.assign(out_data[0], req[0], in_data[0])
        # return
        data = in_data[0]
        if is_train and self.init:
            # in_data[1][:] = mx.nd.min(mx.nd.abs(data))
            # in_data[2][:] = mx.nd.max(mx.nd.abs(data))
            in_data[1][:] = mx.nd.array([0])
            in_data[2][:] = mx.nd.array([1.0])
            in_data[3][:] = mx.nd.array([1.0])
            self.init = False
        pruning_point = in_data[1]
        clipping_point = in_data[2]
        gamma = in_data[3]

        self.max = mx.nd.max(mx.nd.abs(data))
        data = data / self.max

        
        pruning = pruning_point.asnumpy()
        clipping = clipping_point.asnumpy()
        if pruning[0] < 0:
            # print("bad pruning < 0, {}".format(pruning[0]) )
            in_data[1][:] = mx.nd.zeros_like(in_data[1])[:]
            pruning_point = in_data[1]
        # else:
        #     print("great pruning >= 0. {}".format(pruning[0]))
        if clipping[0] > 1:
            # print("bad clipping > 1, {}".format(clipping[0]))
            in_data[2][:] = mx.nd.ones_like(in_data[2])[:]
            clipping_point = in_data[2]
        # else:
        #     print("great clipping <= 1.{}".format(clipping[0]))

        center = 0.5 * (clipping_point + pruning_point)
        distance = 0.5 * (clipping_point - pruning_point)
        alpha = 0.5 / distance
        beta = -0.5 * center / distance + 0.5

        data_abs = mx.nd.abs(data)
        data_sign = mx.nd.sign(data)
        interval_data = data_abs * (data_abs > pruning_point) * (data_abs < clipping_point)
        if self.is_weight:
            transformed_data = data_sign * (data_abs > clipping_point) + \
                               data_sign * (alpha * interval_data + beta)
            # transformed_data = data_sign * (data_abs > clipping_point) + \
            #                    data_sign * mx.nd.power((alpha * interval_data + beta), gamma)
            self.assign(out_data[0], req[0], (mx.nd.round(transformed_data * self.QUANT_LEVEL) / 
                                                  self.QUANT_LEVEL) )
        else:
            transformed_data = data_sign * (data_abs > clipping_point) + \
                               data_sign * (alpha * interval_data + beta)
            self.assign(out_data[0], req[0], (mx.nd.round(transformed_data * self.QUANT_LEVEL) / 
                                                 self.QUANT_LEVEL) )

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.count += 1
        assert len(req) >= 3
        assert self.fix_gamma == True, "currently only support fix gamma"

        data = in_data[0]
        pruning_point = in_data[1]
        clipping_point = in_data[2]
        gamma = in_data[3]

        data = data / self.max
        data_abs = mx.nd.abs(data)
        data_sign = mx.nd.sign(data)
        interval_flag = (data_abs > pruning_point) * (data_abs < clipping_point)
        
        interval_grad = interval_flag * out_grad[0]
        # data = interval_flag * data # there is no need to get interval data

        pruning_grad = mx.nd.sum(interval_grad * ( (data - clipping_point* data_sign) /
                                                  ((clipping_point - pruning_point)**2) ) )
        clipping_grad = mx.nd.sum(interval_grad * (- (data - pruning_point * data_sign) / 
                                                    ((clipping_point - pruning_point)**2) ) )

        self.assign(in_grad[0], req[0], interval_grad / (clipping_point - pruning_point) / self.max)
        # print("pruning grad:{}, clipping_grad:{}, pruning:{}, clipping:{}, sum(interval_grad):{}".format(
        #       pruning_grad.asnumpy()[0], clipping_grad.asnumpy()[0], 
        #       pruning_point.asnumpy()[0], clipping_point.asnumpy()[0],
        #       mx.nd.sum(interval_grad).asnumpy()[0]))
        self.assign(in_grad[1], req[1], pruning_grad)
        self.assign(in_grad[2], req[2], clipping_grad)
        
@mx.operator.register("QIL_PY")
class QIL_PYProp(mx.operator.CustomOpProp):
    def __init__(self, is_weight="False", fix_gamma="True", nbits="4"):
        self.is_weight = eval(is_weight)
        self.fix_gamma = eval(fix_gamma)
        self.nbits = int(nbits)
        super(QIL_PYProp, self).__init__(True)
    def list_arguments(self):
        return ['data', 'pruning_point', 'clipping_point', 'gamma']
    def list_outputs(self):
        return ['output']
    def list_auxiliary_states(self):
        return []
    def infer_shape(self, in_shape):
        shape = in_shape[0]
        return [shape, [1], [1], [1]], [shape], []
    def infer_type(self, in_type):
        dtype = in_type[0]
        return [dtype] * len(in_type), [in_type[0]]*len(self.list_outputs()), \
            [in_type[0]]*len(self.list_auxiliary_states())

    def create_operator(self, ctx, shapes, dtypes):
        return QIL_PY(self.is_weight, self.fix_gamma, self.nbits)
