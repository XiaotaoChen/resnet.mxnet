import mxnet as mx
import numpy as np
import copy
import math


def print_info(auto_grad, cal_grad, name):
    print("{} autograd:\n{}\n cal grad:\n{}".format(name, auto_grad.asnumpy(), cal_grad.asnumpy()))
    print("{} autograd - cal_grad:\n{}".format(name, auto_grad.asnumpy() - cal_grad.asnumpy()))

def simulate_backward(data, gamma, out_grad, is_weight, grad_factor):
    clip_flag = (data < gamma)
    if is_weight:
        clip_flag = clip_flag * (data > - gamma)
    data_grad = out_grad * clip_flag
    outlier_grad = np.sum(out_grad * ((1 - clip_flag) * np.sign(data)) )
    inner_grad = np.sum(out_grad * (np.round(data / gamma) - data / gamma) * clip_flag)
    gamma_grad = outlier_grad + inner_grad
    if grad_factor:
        num_data = data.size
        gamma_grad = gamma_grad / math.sqrt(num_data)
    return data_grad, outlier_grad, inner_grad, gamma_grad

class LSQ_PY(mx.operator.CustomOp):
    def __init__(self, nbits, is_weight, grad_factor):
        self.nbits = nbits
        self.is_weight = is_weight
        self.grad_factor = grad_factor
        self.QUANT_LEVEL = 2**self.nbits -1
        self.count=0

        self.data = None
        self.gamma = None
        self.output = None

    def forward(self, is_train, req, in_data, out_data, aux):
        # self.assign(out_data[0], req[0], in_data[0])
        # return
        assert len(in_data) == 2, "the input must be 2 in PACT: data and gamma"
        self.data = in_data[0]
        self.gamma = in_data[1]
        self.data.attach_grad()
        self.gamma.attach_grad()
        # print("{} gamma:{}".format(self.count, gamma.asnumpy()[0]))
        # self.count += 1
        # old_cliped = mx.nd.clip(self.data, 0, self.gamma.asnumpy()[0])

        with mx.autograd.record():
            self.output = mx.nd.where(self.data < self.gamma, self.data, self.gamma.broadcast_like(self.data))
            if self.is_weight:
                self.output = mx.nd.where(self.output > (- self.gamma), self.output, - self.gamma.broadcast_like(self.data))

        quant_unit = self.gamma / self.QUANT_LEVEL
        self.assign(out_data[0], req[0], mx.nd.round(self.output / quant_unit) * quant_unit)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        # self.assign(in_grad[0], req[0], out_grad[0])
        # return

        # cliped_flag = data >= gamma
        # # gamma_grad = 1 if data >= gamma
        # gamma_grad = mx.nd.sum(cliped_flag)
        # data = in_data[0]
        # gamma = in_data[1]
        # self.assign(in_grad[0], req[0], out_grad[0] * (data < gamma))
        # self.assign(in_grad[1], req[1], mx.nd.sum(out_grad[0] * (data >= gamma)))

        self.output.backward(out_grad[0])
        self.assign(in_grad[0], req[0], self.data.grad)
        # self.assign(in_grad[1], req[1], self.gamma.grad)
        # gamma_grad = outiler_grad + inner_grad
        clip_flag = (self.data < self.gamma)
        if self.is_weight:
            clip_flag = clip_flag * (self.data > - self.gamma)
        inners_grad = mx.nd.sum(out_grad[0] * (mx.nd.round(self.data / self.gamma) - self.data / self.gamma) * clip_flag)
        gamma_grad = inners_grad + self.gamma.grad
        if self.grad_factor:
            gamma_grad = gamma_grad / math.sqrt(self.data.size)
        self.assign(in_grad[1], req[1], gamma_grad)
        # print("outlier grad:{}, inner grad:{}, gamma grad:{}, data.size:{}".format(self.gamma.grad.asnumpy()[0], 
        #                                                              inners_grad.asnumpy()[0],
        #                                                              in_grad[1].asnumpy()[0],
        #                                                              self.data.size))


        # print("data:\n{}".format(self.data))
        # print("gamma:\n{}".format(self.gamma))
        # print("out grad:\n{}".format(out_grad[0]))
        # print("clip flag:\n{}".format(clip_flag))

        # print("data grad:\n{}".format(in_grad[0]))
        # print("out grad:\n{}".format(self.gamma.grad))
        # print("inner grad:\n{}".format(inners_grad))
        # print("total grad:\n{}".format(gamma_grad))
        # sim_data_grad, sim_out_grad, sim_inn_grad, total_grad = simulate_backward(self.data.asnumpy(), 
        #                                                               self.gamma.asnumpy(), 
        #                                                               out_grad[0].asnumpy(), 
        #                                                               self.is_weight, 
        #                                                               self.grad_factor)
        # print("simulate data grad:\n{}".format(sim_data_grad))
        # print("simulate out grad:\n{}".format(sim_out_grad))
        # print("simulate inner grad:\n{}".format(sim_inn_grad))
        # print("simulate total grad:\n{}".format(total_grad))
  
@mx.operator.register("LSQ_PY")
class LSQ_PYProp(mx.operator.CustomOpProp):
    def __init__(self, nbits="8", is_weight="False", grad_factor="False"):
        self.nbits = eval(nbits)
        self.is_weight = eval(is_weight)
        self.grad_factor = eval(grad_factor)
        super(LSQ_PYProp, self).__init__(True)
    def list_arguments(self):
        return ['data', "gamma"]
    def list_outputs(self):
        return ['output']
    def list_auxiliary_states(self):
        return []
    def infer_shape(self, in_shape):
        shape = in_shape[0]
        return [shape, [1]], [shape], []
    def infer_type(self, in_type):
        dtype = in_type[0]
        return [dtype] * len(in_type), [in_type[0]]*len(self.list_outputs()), \
            [in_type[0]]*len(self.list_auxiliary_states())

    def create_operator(self, ctx, shapes, dtypes):
        return LSQ_PY(self.nbits, self.is_weight, self.grad_factor)

