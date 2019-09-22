import mxnet as mx
import numpy as np
import copy

class ClipGrad_Quantization_int8(mx.operator.CustomOp):
    def __init__(self, quant_mode, is_weight, is_weight_perchannel, delay_quant, ema_decay):
        self.quant_mode = quant_mode
        self.is_weight = is_weight
        self.is_weight_perchannel = is_weight_perchannel
        self.delay_quant = delay_quant
        self.ema_decay = ema_decay
        self.QUANT_LEVEL = 127
        self.init = True
    def forward(self, is_train, req, in_data, out_data, aux):
        if is_train and self.delay_quant > 0:
            self.assign(out_data[0], req[0], in_data[0])
            self.delay_quant -= 1
            return
        if self.is_weight:
            if self.is_weight_perchannel:
                # save weight maxs
                if is_train > 0:
                    data = mx.nd.abs(in_data[0])
                    reduce_axis = tuple([i for i in range(len(data.shape))])
                    maxs = mx.nd.max(data, axis=reduce_axis[1:])
                    aux[0][:] = maxs
                target_shape = (in_data[0].shape[0],) + (1,) * len(in_data[0].shape[1:])
                quant_unit = aux[0] / self.QUANT_LEVEL
                quant_unit = quant_unit.reshape(target_shape).broadcast_like(in_data[0])
            else:
                if is_train > 0:
                    data = mx.nd.abs(in_data[0])
                    maxs = mx.nd.max(data)
                    aux[0][:] = maxs
                quant_unit = aux[0] / self.QUANT_LEVEL
            self.assign(out_data[0], req[0], mx.nd.round(in_data[0] / quant_unit) * quant_unit)
        else:
            if is_train:
                data = mx.nd.abs(in_data[0])
                maxs = mx.nd.max(data)
                # udpate activation maxs
                if self.init:
                    aux[0][:] = maxs
                    self.init = False
                else:
                    aux[0][:] = aux[0] * self.ema_decay + maxs * (1 - self.ema_decay)
            quant_unit = aux[0] / self.QUANT_LEVEL
            out_data[0][:] = mx.nd.clip(in_data[0], 
                                        - aux[0].asnumpy()[0], 
                                        aux[0].asnumpy()[0])
            out_data[0][:] = mx.nd.round(out_data[0] / quant_unit) * quant_unit
            # out_data[0][:] = mx.nd.clip(mx.nd.round(in_data[0] / quant_unit), - self.QUANT_LEVEL, self.QUANT_LEVEL)
            # out_data[0][:] = out_data[0][:] * quant_unit

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        # for weight quantize: no need to clip, pass all grad straightforward
        # for act quantize: only pass grad within clip range
        if self.is_weight:
            self.assign(in_grad[0], req[0], out_grad[0])
        else:
            in_grad[0][:] = out_grad[0]
            # assign 0 to the in data whose value is out of the clip range
            # ndarray don't supoprt boolean array indexing
            # assign 0 to the index whose value is less than -aux[0][0]
            in_grad[0][:] = in_grad[0] * (in_data[0] > -aux[0][0])
            # assign 0 to the index which value is more than aux[0][0] 
            in_grad[0][:] = in_grad[0] * (in_data[0] < aux[0][0])
        

@mx.operator.register("ClipGrad_Quantization_int8")
class ClipGradQuantizationInt8Prop(mx.operator.CustomOpProp):
    def __init__(self, quant_mode, is_weight, is_weight_perchannel=False, delay_quant=0, ema_decay=0.99):
        self.quant_mode = str(quant_mode)
        self.delay_quant = int(delay_quant)
        self.ema_decay = float(ema_decay)
        self.is_weight = eval(is_weight)
        self.is_weight_perchannel = eval(is_weight_perchannel)
        super(ClipGradQuantizationInt8Prop, self).__init__(True)
    def list_arguments(self):
        return ['data']
    def list_outputs(self):
        return ['output']
    def list_auxiliary_states(self):
        return ["minmax"]
    def infer_shape(self, in_shape):
        shape = in_shape[0]
        if self.is_weight_perchannel and self.is_weight:
            aux_shape = [shape[0]]
        else:
            aux_shape = [1]
        return [shape], [shape], [aux_shape]
    def infer_type(self, in_type):
        return in_type, in_type, in_type 

    def create_operator(self, ctx, shapes, dtypes):
        return ClipGrad_Quantization_int8(self.quant_mode, self.is_weight,
                                 self.is_weight_perchannel,
                                 self.delay_quant, self.ema_decay)