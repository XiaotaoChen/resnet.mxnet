import mxnet as mx
import numpy as np
import copy



def print_info(auto_grad, cal_grad, name):
    print("{} autograd:\n{}\n cal grad:\n{}".format(name, auto_grad.asnumpy(), cal_grad.asnumpy()))
    print("{} autograd - cal_grad:\n{}".format(name, auto_grad.asnumpy() - cal_grad.asnumpy()))

def interval_quantize(data, pruning_point, clipping_point, quant_level):
    data_abs = mx.nd.abs(data)
    data_zeros = data == 0
    interval = (clipping_point - pruning_point) / quant_level
    quant_data = mx.nd.round( (data - pruning_point) / interval ) * interval + pruning_point
    # only quant the interval data, resetting 0 to the lower range of pruning_point
    quant_data[data_zeros] = 0
    return quant_data


class QIL_V3_PY(mx.operator.CustomOp):
    def __init__(self, is_weight, fix_gamma, nbits):
        self.is_weight = is_weight
        self.fix_gamma = fix_gamma
        self.nbits = nbits
        self.QUANT_LEVEL = 2**self.nbits -1


        self.data = None
        self.ep = None
        self.ed = None
        self.output = None

        self.count = 0

    def forward(self, is_train, req, in_data, out_data, aux):
        self.data = in_data[0]
        self.ep = in_data[1]
        self.ed = in_data[2]
        gamma = in_data[3]

        self.data.attach_grad()
        self.ep.attach_grad()
        self.ed.attach_grad()
        
        # print("count:{}, pruning:{}, clippig:{}".format(self.count, self.pruning_point.asnumpy()[0], self.clipping_point.asnumpy()[0]))
        # self.count += 1

        with mx.autograd.record():
            pruning_point = mx.nd.exp(self.ep)
            distance = mx.nd.exp(self.ed)
            clipping_point = pruning_point + distance

            data_abs = mx.nd.abs(self.data)
            data_sign = mx.nd.sign(self.data)
            interval_flag = (data_abs >= pruning_point) * (data_abs <= clipping_point)

            self.output = data_sign * (data_abs > clipping_point) + \
                        data_sign * ( (data_abs - pruning_point) / distance) * interval_flag
            
        self.assign(out_data[0], req[0], mx.nd.round(self.output * self.QUANT_LEVEL) / self.QUANT_LEVEL)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        assert len(req) >= 3
        assert self.fix_gamma == True, "currently only support fix gamma"

        self.output.backward(out_grad[0])
        self.assign(in_grad[0], req[0], self.data.grad)
        self.assign(in_grad[1], req[1], self.ep.grad)
        self.assign(in_grad[2], req[2], self.ed.grad)

        # data = in_data[0]
        # ep = in_data[1]
        # ed = in_data[2]
        # gamma = in_data[3]

        # pruning_point = mx.nd.exp(ep)
        # distance = mx.nd.exp(ed)
        # clipping_point = pruning_point + distance

        # data_abs = mx.nd.abs(data)
        # data_sign = mx.nd.sign(data)
        # interval_flag = (data_abs >= pruning_point) * (data_abs <= clipping_point)
        
        # data_grad = out_grad[0] / distance * interval_flag

        # ep_grad = mx.nd.sum(out_grad[0] * interval_flag * \
        #                     ((- 1/distance) * data_sign * pruning_point))
        # ed_grad = mx.nd.sum(out_grad[0] * interval_flag * \
        #                     (-(data - pruning_point * data_sign) / distance))

        # print("data:\n{}\n ep:{}, ed:{}, pruning_point:{}, clipping_point:{}".format(data, \
        #                                         ep.asnumpy()[0], ed.asnumpy()[0], \
        #                                         pruning_point.asnumpy()[0], clipping_point.asnumpy()[0]))
        # print_info(self.data.grad, data_grad, "data grad")
        # print_info(self.ep.grad, ep_grad, "ep grad")
        # print_info(self.ed.grad, ed_grad, "ed grad")
        
@mx.operator.register("QIL_V3_PY")
class QIL_V3_PYProp(mx.operator.CustomOpProp):
    def __init__(self, is_weight="False", fix_gamma="True", nbits="4"):
        self.is_weight = eval(is_weight)
        self.fix_gamma = eval(fix_gamma)
        self.nbits = int(nbits)
        super(QIL_V3_PYProp, self).__init__(True)
    def list_arguments(self):
        return ['data', 'ep', 'ed', 'gamma']
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
        return QIL_V3_PY(self.is_weight, self.fix_gamma, self.nbits)
