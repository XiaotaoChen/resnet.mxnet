import mxnet as mx
import numpy as np
import copy

def simulate(data, is_weight, is_weight_perchannel, old_threshold=None):
    data_abs = np.abs(data)
    if is_weight:
        if is_weight_perchannel:
            channel = data.shape[0]
            thresholds = np.max(data_abs, axis=(1,2,3))
            thresholds = thresholds.reshape((channel, 1, 1, 1))
        else:
            thresholds = np.max(data_abs)
        quant_unit = thresholds / 127
        quant_data = np.round(data / quant_unit) * quant_unit
    else:
        thresholds = np.max(data_abs)
        thresholds = 0.99 * old_threshold + (1 - 0.99) * thresholds
        data = np.clip(data, -thresholds, thresholds)
        quant_unit = thresholds / 127
        quant_data = np.round(data /quant_unit) * quant_unit
    return quant_data

class Quantization_int8(mx.operator.CustomOp):
    def __init__(self, 
                 nbits, 
                 quant_mode, 
                 is_weight, 
                 is_weight_perchannel, 
                 delay_quant, 
                 ema_decay, 
                 grad_mode, 
                 fix_act_scale):
        self.nbits = nbits
        self.quant_mode = quant_mode
        self.is_weight = is_weight
        self.is_weight_perchannel = is_weight_perchannel
        self.delay_quant = delay_quant
        self.ema_decay = ema_decay
        self.grad_mode = grad_mode
        self.fix_act_scale = fix_act_scale
        self.QUANT_LEVEL = 2**self.nbits -1
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
            # self.assign(out_data[0], req[0], mx.nd.round(in_data[0] / quant_unit) * quant_unit)

        # if self.is_weight is False:
        #     simulate_quant = simulate(in_data[0].asnumpy(), True, self.is_weight_perchannel)
        #     print("aux:{}".format(aux[0].asnumpy()))
        #     print("in data:{}".format(in_data[0].asnumpy()))
        #     print("forward quant:{}".format(out_data[0].asnumpy()))
        #     print("simulate quant:{}".format(simulate_quant))
        #     print("simulate - forward:{}".format(simulate_quant - out_data[0].asnumpy()))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], out_grad[0])

@mx.operator.register("Quantization_int8_PY")
class QuantizationInt8Prop(mx.operator.CustomOpProp):
    def __init__(self, 
                 nbits, 
                 quant_mode, 
                 is_weight, 
                 is_weight_perchannel=False, 
                 delay_quant=0, 
                 ema_decay=0.99, 
                 grad_mode="ste", 
                 fix_act_scale=False):
        self.nbits = int(nbits)
        self.quant_mode = str(quant_mode)
        self.is_weight = eval(is_weight)
        self.is_weight_perchannel = eval(is_weight_perchannel)
        self.delay_quant = int(delay_quant)
        self.ema_decay = float(ema_decay)
        self.grad_mode = str(grad_mode)
        self.fix_act_scale = eval(fix_act_scale)
        
        super(QuantizationInt8Prop, self).__init__(True)
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
        return Quantization_int8(self.nbits, self.quant_mode, self.is_weight, self.is_weight_perchannel,
                                 self.delay_quant, self.ema_decay, self.grad_mode, self.fix_act_scale)

class Fold_BN(mx.operator.CustomOp):
    def __init__(self, quant_mode, is_weight_perchannel, delay_quant, ema_decay,
                 name, num_filter, num_group, kernel, stride, pad, dilate, no_bias, 
                 eps, momentum, fix_gamma, total_params_path, params_prefix,
                 quantize_flag):
        self.quant_mode = quant_mode
        self.is_weight_perchannel = is_weight_perchannel
        self.delay_quant = delay_quant
        self.ema_decay = ema_decay
        self.QUANT_LEVEL = 127
        self.init = True
        # conv params
        self.name = name
        self.num_filter = num_filter
        self.num_group = num_group
        self.kernel = kernel
        self.stride = stride
        self.pad = pad
        self.dilate = dilate
        self.no_bias = no_bias
        # bn params
        self.eps = eps
        self.momentum = momentum
        self.fix_gamma = fix_gamma
        # for inference
        self.total_params_path = total_params_path
        import os
        self.params_prefix = params_prefix
        if os.path.exists(self.total_params_path):
            params = mx.nd.load(self.total_params_path)
            self.mean = params["aux:{}_batchnorm_moving_mean".format(self.params_prefix)
                              ].as_in_context(mx.gpu(0))
            self.var = params["aux:{}_batchnorm_moving_var".format(self.params_prefix)
                             ].as_in_context(mx.gpu(0))
        else:
            self.mean = None
            self.var = None
        # for debug
        self.quantize_flag = quantize_flag
        # print("[added by cxt] quantize flag:{}, params_prefix:{}, params_path:{}".format(
        #       self.quantize_flag, self.params_prefix, self.total_params_path))

    def forward(self, is_train, req, in_data, out_data, aux):
        assert len(in_data) == 7, "fold bn require six inputs: data, weight, bn_input, bn_gamma, bn_beta, bn_mean, bn_var"
        data = in_data[0]
        weight = in_data[1]
        bn_gamma = in_data[3]
        bn_beta = in_data[4]
        bn_mean = in_data[5]
        bn_var = in_data[6]

        # print("bn_mean:{}\n bn_var:{}".format(bn_mean, bn_var))

        if is_train and self.delay_quant > 0:
            # assign bn output to output
            self.assign(out_data[0], req[0], in_data[2])
            self.delay_quant -= 1
            return
        """
        in train mode, the bn_var seems like wrong, we should to forward convolution to calculate the conv_var;
        """
        conv = mx.nd.Convolution(
            name=self.name,
            data=data,
            num_filter=self.num_filter,
            kernel=self.kernel,
            num_group=self.num_group,
            stride=self.stride,
            pad=self.pad,
            dilate=self.dilate,
            no_bias=self.no_bias,
            weight=weight
        )
        if is_train:
            # in training mode, bn_var is not correct from BatchNorm
            bn_var = mx.nd.mean(mx.nd.square(conv - bn_mean.reshape(1,conv.shape[1],1,1)), axis=(0,2,3))
        else:
            # in inference mode, bn_mean, bn_mean both aren't correct from BatchNorm, We should read from the params file
            assert self.mean is not None and self.var is not None, "in \
                   inference mode must offer mean,var to avoid the BatchNorm bug"
            assert bn_mean.shape == self.mean.shape and \
                   bn_var.shape == self.var.shape, "{} the mean or var shape \
                   is not match".format(self.name)
            bn_mean = self.mean.as_in_context(bn_mean.context)
            bn_var = self.var.as_in_context(bn_var.context)
        
        # check_fold_bn_consistence(bn_output=in_data[2], data=in_data[0], weight=weight,
        #                           bn_gamma=bn_gamma, bn_beta=bn_beta, bn_mean=bn_mean, bn_var=bn_var,
        #                           num_filter=self.num_filter, kernel=self.kernel, num_group=self.num_group, 
        #                           stride=self.stride, pad=self.pad, dilate=self.dilate, no_bias=self.no_bias)

        if self.quantize_flag:
            # quantize input
            if is_train:
                data_abs = mx.nd.abs(in_data[0])
                maxs = mx.nd.max(data_abs)
                # udpate acativation maxs
                if self.init:
                    aux[0][:] = maxs
                    self.init = False
                else:
                    aux[0][:] = aux[0] * self.ema_decay + maxs * (1 - self.ema_decay)
            quant_unit = aux[0] / self.QUANT_LEVEL
            data = mx.nd.round(data / quant_unit) * quant_unit


        w_target_shape = (weight.shape[0],) + (1,) * len(weight.shape[1:])
        # flod bn to multip gamma/sqrt(var + eps)
        factor = bn_gamma / mx.nd.sqrt(bn_var + self.eps)
        factor = factor.reshape(w_target_shape)
        weight = weight * factor

        if self.quantize_flag:
            # quantize weight
            weight_abs = mx.nd.abs(weight)
            if self.is_weight_perchannel:
                reduce_axis = tuple([i for i in range(len(weight.shape))])
                maxs = mx.nd.max(weight_abs, axis=reduce_axis[1:])
                quant_unit = maxs / self.QUANT_LEVEL
                quant_unit = quant_unit.reshape(w_target_shape).broadcast_like(weight)
            else:
                maxs = mx.nd.max(weight_abs)
                quant_unit = maxs / self.QUANT_LEVEL
            if is_train:
                aux[1][:] = maxs
            weight = mx.nd.round(weight / quant_unit) * quant_unit
        
        # conv
        conv = mx.nd.Convolution(
            name=self.name,
            data=data,
            num_filter=self.num_filter,
            kernel=self.kernel,
            num_group=self.num_group,
            stride=self.stride,
            pad=self.pad,
            dilate=self.dilate,
            no_bias=self.no_bias,
            weight=weight
        )

        # flod bn to add beta -  gamma* mean/sqrt(var + eps)
        bias = bn_beta -  bn_mean * bn_gamma / mx.nd.sqrt(bn_var + self.eps)
        # bias = bn_beta -  bn_mean * factor # this method will cause error.
        target_shape = (1, conv.shape[1], 1, 1)
        bias = bias.reshape(target_shape)
        # fold_bn_result = conv + bias
        # print("{} flod bn err:{}".format(self.params_prefix, np.linalg.norm(fold_bn_result.asnumpy() - in_data[2].asnumpy())))

        self.assign(out_data[0], req[0], conv + bias)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        # assign out grad to the output of bn and assign others to 0
        for i in range(len(in_data)):
            in_grad[i][:] = 0
            # self.assign(in_grad[i], req[i], 0)
            # print("{}:{},{}, sum:{}".format(i, in_data[i].shape, in_grad[i].shape, mx.nd.sum(in_grad[i])))
        self.assign(in_grad[2], req[2], out_grad[0])

@mx.operator.register("Fold_BN")
class FoldBNProp(mx.operator.CustomOpProp):
    def __init__(self, quant_mode, is_weight_perchannel=False, delay_quant=0, ema_decay=0.99,
                 name='fold_bn', num_filter=None, num_group=None, kernel=(3,3), stride=(1,1), 
                 pad=(0,0), dilate=(1,1), no_bias=True, eps=1e-5, momentum=0.9, 
                 fix_gamma=False, total_params_path="None",params_prefix="None", quantize_flag="True"):
        self.quant_mode = str(quant_mode)
        self.delay_quant = int(delay_quant)
        self.ema_decay = float(ema_decay)
        self.is_weight_perchannel = eval(is_weight_perchannel)
        # conv params
        self.name = str(name)
        self.num_filter = int(num_filter)
        self.num_group = int(num_group)
        self.kernel = eval(kernel)
        self.stride = eval(stride)
        self.pad = eval(pad)
        self.dilate = eval(dilate)
        self.no_bias = eval(no_bias)
        # bn params
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.fix_gamma = eval(fix_gamma)
        self.total_params_path = str(total_params_path)
        self.params_prefix = params_prefix
        # for debug
        self.quantize_flag = eval(quantize_flag)


        super(FoldBNProp, self).__init__(True)
    def list_arguments(self):
        return ['data', 'weight', 'bn_output', 'bn_gamma', 'bn_beta',  'bn_mean', 'bn_var']
    def list_outputs(self):
        return ['output']
    def list_auxiliary_states(self):
        return ["data_minmax", "weight_minmax"]
    def infer_shape(self, in_shape):
        shape = in_shape[0]
        aux_shape = [[1]]
        if self.is_weight_perchannel:
            # weight shape
            aux_shape.append([in_shape[1][0]])
        else:
            aux_shape.append([1])
        # the batch size
        oshape = [None] * len(shape)
        oshape[0] = shape[0]
        # number of filter
        oshape[1] = self.num_filter

        oshape[2] = int((shape[2] + 2 * self.pad[0] -
            (self.dilate[0] * (self.kernel[0] - 1) + 1)) / self.stride[0] + 1)
        oshape[3] = int((shape[3] + 2 * self.pad[1] -
            (self.dilate[1] * (self.kernel[1] - 1) + 1)) / self.stride[1] + 1)
        return in_shape, [oshape], aux_shape
    def infer_type(self, in_type):
        return in_type, [in_type[0]]*len(self.list_outputs()), \
            [in_type[0]]*len(self.list_auxiliary_states())
    def create_operator(self, ctx, shapes, dtypes):
        return Fold_BN(self.quant_mode, self.is_weight_perchannel, self.delay_quant, self.ema_decay,
                       self.name, self.num_filter, self.num_group, self.kernel, self.stride, self.pad, 
                       self.dilate, self.no_bias, self.eps, self.momentum, self.fix_gamma, 
                       self.total_params_path, self.params_prefix,
                       self.quantize_flag)
