import mxnet as mx
import numpy as np
import copy


def assert_all(pruning_point, clipping_point):
    pruning = pruning_point.asnumpy()
    clipping = clipping_point.asnumpy()
    assert np.all(pruning >= 0), "pruning {} must greater than 0".format(pruning[0])
    assert np.all(pruning < clipping), "pruning vs clipping {} vs {} pruning must less \
        than clipping".format(pruning[0], clipping[0])
    assert np.all(clipping <= 1), "clipping {} must less than 1.0".format(clipping[0])

def interval_quantize(interval_data, sign_data, pruning_point, clipping_point, quant_level):
    interval = (clipping_point - pruning_point) / quant_level
    return mx.nd.round( (interval_data - pruning_point * sign_data) / interval ) * interval + pruning_point * sign_data

class QIL_V2_PY(mx.operator.CustomOp):
    def __init__(self, is_weight, fix_gamma, nbits):
        self.is_weight = is_weight
        self.fix_gamma = fix_gamma
        self.nbits = nbits
        self.QUANT_LEVEL = 2**self.nbits -1

        self.max = 0

        self.data = None
        self.center = None
        self.distance = None
        self.gamma = None
        self.output = None

    def forward(self, is_train, req, in_data, out_data, aux):
        assert len(in_data) == 4, "the input must be 4 in QIL_V2: data, center, distance, gamma"
        self.data = in_data[0]
        self.center = in_data[1]
        self.distance = in_data[2]
        self.gamma = in_data[3]

        self.data.attach_grad()
        self.center.attach_grad()
        self.distance.attach_grad()
        self.gamma.attach_grad()

        with mx.autograd.record():
            clipping_point = self.center + self.distance
            pruning_point = self.center - self.distance
            alpha = 0.5 / self.distance
            beta = -0.5 * self.center / self.distance + 0.5
            data_abs = mx.nd.abs(self.data)
            data_sign = mx.nd.sign(self.data)
            # data_sign = (self.data > 0) + ((self.data > 0) - 1)
            # interval_data = data_abs * (data_abs > pruning_point) * (data_abs < clipping_point)
            interval_flag =(data_abs >= pruning_point) * (data_abs <= clipping_point)
            self.output = data_sign * (data_abs > clipping_point) + \
                               data_sign * (alpha * data_abs + beta) * interval_flag
        
        self.assign(out_data[0], req[0], \
                    mx.nd.round(self.output * self.QUANT_LEVEL) / self.QUANT_LEVEL)


        # print("out_data:\n{}\nreq:{}".format(out_data[0], req[0]))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        assert len(req) >= 3
        assert self.fix_gamma == True, "currently only support fix gamma"

        self.output.backward(out_grad[0])
        self.assign(in_grad[0], req[0], self.data.grad)
        self.assign(in_grad[1], req[1], self.center.grad)
        self.assign(in_grad[2], req[2], self.distance.grad)

        # data = in_data[0]
        # center = in_data[1]
        # distance = in_data[2]
        # gamma = in_data[3]

        # pruning_point = center - distance
        # clipping_point = center + distance

        # data_abs = mx.nd.abs(data)
        # data_sign = mx.nd.sign(data) # when data is 0, the sign with be 0. this won't affect the results
        # # data_sign = (data > 0) + ((data > 0) - 1)
        # interval_flag = (data_abs >= pruning_point) * (data_abs <= clipping_point)
        
        # interval_grad = interval_flag * out_grad[0]
        # data = interval_flag * data
        # # data_sign =  interval_flag * data_sign

        # data_grad = interval_grad * (0.5 / distance)
        # center_grad = mx.nd.sum(interval_grad * ( - 0.5 / distance * data_sign ) )
        # distance_grad = mx.nd.sum(interval_grad * ( -0.5 * (data - center * data_sign) / (distance**2) ) )
        # print("data:\n{}\n center:{}, distance:{}".format(data, center.asnumpy()[0], distance.asnumpy()[0]))
        # print("out grad:\n{}\nsign:\n{}\n interval grad:\n{}".format(out_grad[0], data_sign, interval_grad))
        
        # print("-0.5* (data - center * sign):\n{}\n distance_grad:\n{}".format(-0.5 * (data - center * data_sign),
        #       interval_grad * ( -0.5 * (data - center * data_sign) ) ))


        # print("distance.grad:\n{}\n cal distance grad:\n{}".format(self.distance.grad, distance_grad))
        # print("distance.grad - cal_grad:\n{}".format(self.distance.grad - distance_grad))

        # self.assign(in_grad[0], req[0], interval_grad * (0.5 / distance) / self.max)
        # self.assign(in_grad[1], req[1], center_grad)
        # self.assign(in_grad[2], req[2], distance_grad)
        
@mx.operator.register("QIL_V2_PY")
class QIL_PYProp(mx.operator.CustomOpProp):
    def __init__(self, is_weight="False", fix_gamma="True", nbits="4"):
        self.is_weight = eval(is_weight)
        self.fix_gamma = eval(fix_gamma)
        self.nbits = int(nbits)
        super(QIL_PYProp, self).__init__(True)
    def list_arguments(self):
        return ['data', 'center', 'distance', 'gamma']
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
        return QIL_V2_PY(self.is_weight, self.fix_gamma, self.nbits)
