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
        self.init = True
        self.count = 0

        self.max = 0

    def forward(self, is_train, req, in_data, out_data, aux):
        # self.assign(out_data[0], req[0], in_data[0])
        # return
        data = in_data[0]
        if is_train and self.init:
            in_data[1][:] = mx.nd.array([0.5])
            in_data[2][:] = mx.nd.array([0.5])
            in_data[3][:] = mx.nd.array([1.0])
            self.init = False
        center = in_data[1]
        distance = in_data[2]
        gamma = in_data[3]

        self.max = mx.nd.max(mx.nd.abs(data))
        data = data / self.max

        print("************* count:{} forward ****************".format(self.count))
        center_np = center.asnumpy()
        distance_np = distance.asnumpy()
        if center_np[0] <= 0:
            print("bad center < 0, {}".format(center_np[0]))
            in_data[1][:] = mx.nd.array([0.5])
            center = in_data[1]
        else:
            print("great center > 0, {}".format(center_np[0]))
        center_np = center.asnumpy()
        if distance_np[0]<=0 or distance_np[0] > 0.5 or distance_np[0] > center_np[0]:
            print("bad distance >0.5 or distance > center, {}".format(distance_np[0]))
            in_data[2][:] = mx.nd.array([min(0.5, center_np[0])])
            distance = in_data[2]
        else:
            print("great distance <= 1, {}".format(distance_np[0]))
        clipping_point = center + distance
        pruning_point = center - distance

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
        assert len(req) >= 3
        assert self.fix_gamma == True, "currently only support fix gamma"

        data = in_data[0]
        center = in_data[1]
        distance = in_data[2]
        gamma = in_data[3]

        pruning_point = center - distance
        clipping_point = center + distance

        data = data / self.max
        data_abs = mx.nd.abs(data)
        data_sign = mx.nd.sign(data)
        interval_flag = (data_abs > pruning_point) * (data_abs < clipping_point)
        
        interval_grad = interval_flag * out_grad[0]

        center_grad = mx.nd.sum(interval_grad * ( - 0.5 / distance * data_sign ) )
        distance_grad = mx.nd.sum(interval_grad * ( -0.5 * (data - center * data_sign) ) )

        print("center grad:{}, distance grad:{}, center:{}, distance:{}, sum(interval_grad):{}".format(
              center_grad.asnumpy()[0], distance_grad.asnumpy()[0], 
              center.asnumpy()[0], distance.asnumpy()[0],
              mx.nd.sum(interval_grad).asnumpy()[0]))

        self.assign(in_grad[0], req[0], interval_grad * (0.5 / distance) / self.max)
        self.assign(in_grad[1], req[1], center_grad)
        self.assign(in_grad[2], req[2], distance_grad)

        print("************* count:{} backward ****************".format(self.count))

        self.count += 1



# q_x = 0.5/center * x
class QIL_V2_SIMPLE_PY(mx.operator.CustomOp):
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
            in_data[1][:] = mx.nd.array([0.5])
            in_data[3][:] = mx.nd.array([1.0])
            self.init = False
        center = in_data[1]
        gamma = in_data[3]

        self.max = mx.nd.max(mx.nd.abs(data))
        data = data / self.max

        # print("************* count:{} forward ****************".format(self.count))
        # center_np = center.asnumpy()
        # if center_np[0] <= 0:
        #     print("bad center < 0, {}".format(center_np[0]))
        #     in_data[1][:] = mx.nd.array([0.5])
        #     center = in_data[1]
        # else:
        #     print("great center > 0, {}".format(center_np[0]))
        clipping_point = 2 * center

        alpha = 0.5 / center

        data_abs = mx.nd.abs(data)
        data_sign = mx.nd.sign(data)
        interval_data = data_abs * (data_abs < clipping_point)
        if self.is_weight:
            transformed_data = data_sign * (data_abs > clipping_point) + \
                               data_sign * (alpha * interval_data)
            self.assign(out_data[0], req[0], (mx.nd.round(transformed_data * self.QUANT_LEVEL) / 
                                                  self.QUANT_LEVEL) )
        else:
            transformed_data = data_sign * (data_abs > clipping_point) + \
                               data_sign * (alpha * interval_data)
            self.assign(out_data[0], req[0], (mx.nd.round(transformed_data * self.QUANT_LEVEL) / 
                                                 self.QUANT_LEVEL) )

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        assert len(req) >= 3
        assert self.fix_gamma == True, "currently only support fix gamma"

        data = in_data[0]
        center = in_data[1]
        gamma = in_data[3]

        clipping_point = 2 * center

        data = data / self.max
        data_abs = mx.nd.abs(data)
        interval_flag = (data_abs < clipping_point)
        
        interval_grad = interval_flag * out_grad[0]

        center_grad = mx.nd.sum(interval_grad * ( (- 0.5 / (center**2)) * data) )
        if self.count %100 == 0:
            print("center grad:{}, center:{}, sum(interval_grad):{}".format(
                center_grad.asnumpy()[0], center.asnumpy()[0], 
                mx.nd.sum(interval_grad).asnumpy()[0]))

        self.assign(in_grad[0], req[0], interval_grad * (0.5 / center) / self.max)
        self.assign(in_grad[1], req[1], center_grad)

        # print("************* count:{} backward ****************".format(self.count))
        self.count += 1
  

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
        # return QIL_V2_PY(self.is_weight, self.fix_gamma, self.nbits)
        return QIL_V2_SIMPLE_PY(self.is_weight, self.fix_gamma, self.nbits)
