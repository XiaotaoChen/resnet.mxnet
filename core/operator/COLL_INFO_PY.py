import mxnet as mx
import numpy as np
import copy
import math



class COLL_INFO_PY(mx.operator.CustomOp):
    def __init__(self, is_weight):
        self.iter = 0
        self.is_weight = is_weight

    def forward(self, is_train, req, in_data, out_data, aux):
        self.assign(out_data[0], req[0], in_data[0])
        self.iter += 1
        np_data = in_data[0].asnumpy()
        print("{} forward {}: shape:{}, size:{}, min:{:.6f}, max:{:.6f}, mean:{:.6f}, var:{:.6f}".format(
              "weight" if self.is_weight else "act", self.iter, np_data.shape, 
              np_data.size, np.min(np_data), np.max(np_data), np.mean(np_data), np.var(np_data)))
        

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], out_grad[0])
        np_data = out_grad[0].asnumpy()
        print("{} backward {}: shape:{}, size:{}, min:{:.6f}, max:{:.6f}, mean:{:.6f}, var:{:.6f}".format(
              "weight" if self.is_weight else "act", self.iter, np_data.shape, 
              np_data.size, np.min(np_data), np.max(np_data), np.mean(np_data), np.var(np_data)))
        if self.iter >= 2:
            raise NotImplementedError

@mx.operator.register("COLL_INFO_PY")
class COLL_INFO_PYProp(mx.operator.CustomOpProp):
    def __init__(self, is_weight="False"):
        self.is_weight = eval(is_weight)
        super(COLL_INFO_PYProp, self).__init__(True)
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
        return COLL_INFO_PY(self.is_weight)

