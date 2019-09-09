import mxnet as mx
import numpy as np
import copy

class Quantization_int8(mx.operator.CustomOp):
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
            data = mx.nd.abs(in_data[0])
            if self.is_weight_perchannel:
                target_shape = (data.shape[0],) + (1,) * len(data.shape[1:])
                reduce_axis = tuple([i for i in range(len(data.shape))])
                maxs = mx.nd.max(data, axis=reduce_axis[1:])
                quant_unit = maxs / self.QUANT_LEVEL
                quant_unit = quant_unit.reshape(target_shape).broadcast_like(in_data[0])
            else:
                maxs = mx.nd.max(data)
                quant_unit = maxs / self.QUANT_LEVEL
            self.assign(out_data[0], req[0], mx.nd.round(in_data[0] / quant_unit) * quant_unit)
            # save weight maxs
            if is_train:
                aux[0][:] = maxs
        else:
            if is_train:
                data = mx.nd.abs(in_data[0])
                maxs = mx.nd.max(data)
                # udpate acativation maxs
                aux[0][:] = aux[0] * self.ema_decay + maxs * (1 - self.ema_decay)
            
            quant_unit = aux[0] / self.QUANT_LEVEL
            self.assign(out_data[0], req[0], mx.nd.round(in_data[0] / quant_unit) * quant_unit)
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], out_grad[0])

@mx.operator.register("Quantization_int8_V2")
class QuantizationInt8Prop(mx.operator.CustomOpProp):
    def __init__(self, quant_mode, is_weight, is_weight_perchannel=False, delay_quant=0, ema_decay=0.99):
        self.quant_mode = str(quant_mode)
        self.delay_quant = int(delay_quant)
        self.ema_decay = float(ema_decay)
        self.is_weight = eval(is_weight)
        self.is_weight_perchannel = eval(is_weight_perchannel)
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
        return Quantization_int8(self.quant_mode, self.is_weight,
                                 self.is_weight_perchannel,
                                 self.delay_quant, self.ema_decay)

class Multi_Factor(mx.operator.CustomOp):
    def __init__(self):
        super(Multi_Factor, self).__init__()
    def forward(self, is_train, req, in_data, out_data, aux):
        assert len(in_data) == 3, "multi Factor require three inputs: data, bn_gamma, bn_var"
        factor = in_data[1] / in_data[2]
        target_shape = (in_data[0].shape[0],) + (1,) * len(in_data[0].shape[1:])
        broadcast_factor = factor.reshape(target_shape).broadcast_like(in_data[0])
        self.assign(out_data[0], req[0], in_data[0] * broadcast_factor)
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], out_grad[0])

@mx.operator.register("Multi_Factor")
class MultiFactorProp(mx.operator.CustomOpProp):
    def __init__(self, ):
        super(MultiFactorProp, self).__init__(True)
    def list_arguments(self):
        return ['data', 'bn_gamma', 'bn_var']
    def list_outputs(self):
        return ['output']
    def infer_shape(self, in_shape):
        shape = in_shape[0]
        return in_shape, [shape], []
    def infer_type(self, in_type):
        return in_type, [in_type[0]], []
    def create_operator(self, ctx, shapes, dtypes):
        return Multi_Factor()

class Add_Bias(mx.operator.CustomOp):
    def __init__(self):
        super(Add_Bias, self).__init__()
    def forward(self, is_train, req, in_data, out_data, aux):
        assert len(in_data) == 5, "multi Factor require three inputs: data, bn_gamma, bn_var, bn_mean, bn_var"
        bn_gamma = in_data[1]
        bn_beta = in_data[2]
        bn_mean = in_data[3]
        bn_var = in_data[4]
        bias = bn_beta - bn_gamma * bn_mean / bn_var
        target_shape = (in_data[0].shape[0],) + (1,) * len(in_data[0].shape[1:])
        broadcast_bias = bias.reshape(target_shape).broadcast_like(in_data[0])
        self.assign(out_data[0], req[0], in_data[0] + broadcast_bias)
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], out_grad[0])

@mx.operator.register("Add_Bias")
class AddBiasProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(AddBiasProp, self).__init__(True)
    def list_arguments(self):
        return ['data', 'bn_gamma', 'bn_beta', 'bn_mean', 'bn_var']
    def list_outputs(self):
        return ['output']
    def infer_shape(self, in_shape):
        shape = in_shape[0]
        return in_shape, [shape], []
    def infer_type(self, in_type):
        return in_type, [in_type[0]], []
    def create_operator(self, ctx, shapes, dtypes):
        return Add_Bias()


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
        if is_train > 0:
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
        return ["input_minmax", "weight_minmax"]
    def infer_shape(self, in_shape):
        shape = in_shape[0]
        aux_shape = [[1]]
        if self.is_weight_perchannel:
            aux_shape.append([shape[0]])
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


@mx.init.register
class CustomInit(mx.init.Initializer):
    def __init__(self, data):
        super(CustomInit, self).__init__(data=data)
        import numpy as np
        self.data = np.asarray(data)
    def _init_weight(self, _, arr):
        arr[:] = self.data

def get_sym_output_channel(name, sym, data_shape=(1, 3, 224, 224)):
    _, out_shapes, _ = sym.infer_shape(data=data_shape)
    assert len(out_shapes) == 1, 'the output of sym is not equal to 1'
    # print('sym:{}:{}'.format(name, sym))
    return out_shapes[0][1]

def quant_conv(name, data, num_filter, kernel, stride, pad=(0,0), no_bias=False, dilate=(1,1), num_group=1,
               quant_mod='minmax', delay_quant=0, is_weight_perchannel=False, data_shape=(1,3,224,224)):
    if is_weight_perchannel:
        assert quant_mod == "minmax", "currenet weight perchannel only support minmax node with weight"

    input_channel = get_sym_output_channel(name, data, data_shape=data_shape)
    weight = mx.sym.Variable(name=name + "_weight", shape=(num_filter, input_channel // num_group, kernel[0], kernel[1]), 
                             dtype="float32")
    weight_q = mx.sym.Custom(data=weight, name = name + "_weight_quant", quant_mode=quant_mod, is_weight=True,
                             is_weight_perchannel = is_weight_perchannel, ema_decay=0.99, delay_quant=delay_quant, 
                             op_type="Quantization_int8_V2")
    data_q = mx.sym.Custom(data=data, name = name + "_data_quant", quant_mode=quant_mod, is_weight=False,
                             is_weight_perchannel = False, ema_decay=0.99, delay_quant=delay_quant, 
                             op_type="Quantization_int8_V2")
    conv = mx.symbol.Convolution(
        name=name,
        data=data_q,
        num_filter=num_filter,
        kernel=kernel,
        num_group=num_group,
        stride=stride,
        pad=pad,
        no_bias=no_bias,
        dilate=dilate,
        weight=weight_q
    )
    return conv

def quant_fc(name, data, num_hidden, quant_mod='minmax', delay_quant=0, is_weight_perchannel=False, data_shape=(1,3,224,224)):
    if is_weight_perchannel:
        assert quant_mod == "minmax", "currenet weight perchannel only support minmax node with weight"
    input_channel = get_sym_output_channel(name, data, data_shape=data_shape)
    fc_weight = mx.sym.Variable(name=name +"_weight", shape=(num_hidden, input_channel), dtype="float32")
    fc_q = mx.sym.Custom(fc_weight, name= name + "_weight_quant", is_weight=True, ema_decay=0.99, 
                         delay_quant=delay_quant, quant_mode = quant_mod, is_weight_perchannel=is_weight_perchannel,
                         op_type="Quantization_int8_V2")
    fc_data_q = mx.sym.Custom(data=data, name= name + "_data_quant", is_weight=False, ema_decay=0.99, 
                              delay_quant=delay_quant, quant_mode = quant_mod, is_weight_perchannel=False,
                              op_type="Quantization_int8_V2")
    fc = mx.symbol.FullyConnected(data=fc_data_q, num_hidden=num_hidden, name='fc', weight=fc_q)
    return fc

def quant_deconv(name, data, kernel, stride, pad, num_filter, no_bias=True, cudnn_tune='fastest', 
                 quant_mod='minmax', delay_quant=0, is_weight_perchannel=False, data_shape=(1,3,224,224)):
    if is_weight_perchannel:
        assert quant_mod == "minmax", "currenet weight perchannel only support minmax node with weight"
    input_channel = get_sym_output_channel(name, data, data_shape=data_shape)
    weight = mx.sym.Variable(name=name + "_weight", shape=(input_channel, num_filter,
                                    kernel[0], kernel[1]), dtype="float32")
    weight_q = mx.sym.Custom(data=weight, name = name + "_weight_quant", quant_mode=quant_mod, is_weight=True,
                             is_weight_perchannel = is_weight_perchannel, ema_decay=0.99, delay_quant=delay_quant, 
                             op_type="Quantization_int8_V2")
    data_q = mx.sym.Custom(data=data, name = name + "_data_quant", quant_mode=quant_mod, is_weight=False,
                             is_weight_perchannel = False, ema_decay=0.99, delay_quant=delay_quant, 
                             op_type="Quantization_int8_V2")
    deconv = mx.symbol.Deconvolution(name=name, data=data_q, kernel=kernel,stride=stride,pad=pad,
                                   no_bias=no_bias ,num_filter=num_filter,cudnn_tune=cudnn_tune,
                                   weight=weight_q)
    return deconv


def check_fold_bn_consistence(bn_output, data, weight, bn_gamma, bn_beta, bn_mean, bn_var,
                              num_filter=4, kernel=(3,3), num_group=1, 
                              stride=(1,1), pad=(0,0), dilate=(1,1), no_bias=True):
    name = 'check'
    eps = 1e-5

    w_target_shape = (weight.shape[0],) + (1,) * len(weight.shape[1:])
    factor = bn_gamma / mx.nd.sqrt(bn_var + eps)
    factor = factor.reshape(w_target_shape)
    weight = weight * factor
    conv = mx.nd.Convolution(
            name=name,
            data=data,
            num_filter=num_filter,
            kernel=kernel,
            num_group=num_group,
            stride=stride,
            pad=pad,
            dilate=dilate,
            no_bias=no_bias,
            weight=weight
        )
    
    bias = bn_beta -  bn_gamma * bn_mean / mx.nd.sqrt(bn_var + eps)
    target_shape = (1, conv.shape[1], 1, 1)
    bias = bias.reshape(target_shape)
    fold_bn_result = conv + bias

    # print("bn err:{}".format(np.linalg.norm(fold_bn_result.asnumpy() - bn_output.asnumpy())))
    # raise NotImplementedError


def fold_bn(name, data, 
            # quant params
            quant_mod='minmax', is_weight_perchannel=False, delay_quant=0, ema_decay=0.99,
            # conv params
            num_filter=None, kernel=None, stride=None, pad=(0,0), no_bias=True, dilate=(1,1), num_group=1,
            # bn params
            eps=1e-5, momentum=0.9, fix_gamma=False, use_global_stats=False,
            total_params_path=None,
            #for debug
            quantize_flag=True,
            data_shape=(1,3,224,224)):
    if is_weight_perchannel:
        assert quant_mod == "minmax", "currenet weight perchannel only support minmax node with weight"
    input_channel = get_sym_output_channel(name, data, data_shape=data_shape)
    weight = mx.sym.Variable(name=name + "_conv2d_weight", shape=(num_filter, input_channel // num_group, 
                             kernel[0], kernel[1]), dtype="float32")
    bn_gamma_var = mx.symbol.Variable(name + '_batchnorm_gamma', shape=(input_channel,), dtype="float32")
    bn_beta_var = mx.symbol.Variable(name + '_batchnorm_beta', shape=(input_channel,), dtype="float32")

    # conv + bn
    conv = mx.sym.Convolution(name=name + "_conv2d", data=data, weight=weight, num_filter=num_filter, kernel=kernel, num_group=num_group, 
                              stride=stride, pad=pad, no_bias=no_bias, dilate=dilate)
    bn_output_var, bn_mean_var, bn_var_var = mx.sym.BatchNorm(name=name + "_batchnorm", data=conv, gamma=bn_gamma_var, beta=bn_beta_var, 
                          eps=eps, momentum=momentum, fix_gamma=fix_gamma, output_mean_var=True, use_global_stats=use_global_stats)
    # flod bn
    # the argument `name` seems like can't pass to Custom op, so create new arg: params_prefix
    fold_bn = mx.sym.Custom(name=name + "_fold_bn", data=data, weight=weight, bn_output=bn_output_var, bn_gamma=bn_gamma_var, 
                            bn_beta=bn_beta_var, bn_mean=bn_mean_var, bn_var=bn_var_var, 
                            # quant params
                            quant_mode=quant_mod, is_weight_perchannel = is_weight_perchannel, 
                            delay_quant=0, ema_decay=ema_decay, 
                            # conv params
                            num_filter=num_filter, num_group=num_group, kernel=kernel, stride=stride, pad=pad, 
                            dilate=dilate, no_bias=no_bias,
                            # bn params
                            eps=eps, momentum=momentum, fix_gamma=fix_gamma,
                            total_params_path=total_params_path, params_prefix=name,
                            # for debug
                            quantize_flag=quantize_flag,
                            op_type="Fold_BN")
    return fold_bn
    
    

def quant_conv_test(name, data, num_filter, is_weight_perchannel, w_init_data):
    kernel = (3,3)
    stride = (1,1)
    pad = (1,1)
    dilate = (1,1)
    input_channel = get_sym_output_channel(name, data)
    weight = mx.sym.Variable(name=name + "_weight", shape=(num_filter, input_channel, kernel[0], kernel[1]), 
                             init=CustomInit(data=w_init_data), dtype="float32")
    weight_q = mx.sym.Custom(data=weight, name = name + "_weight_quant", quant_mode="minmax", is_weight=True,
                             is_weight_perchannel = is_weight_perchannel, ema_decay=0.99, delay_quant=0, 
                             op_type="Quantization_int8_V2")
    data_q = mx.sym.Custom(data=data, name = name + "_data_quant", quant_mode="minmax", is_weight=False,
                             is_weight_perchannel = False, ema_decay=0.99, delay_quant=0, 
                             op_type="Quantization_int8_V2")
    conv = mx.symbol.Convolution(
        name=name,
        data=data_q,
        num_filter=num_filter,
        kernel=kernel,
        stride=stride,
        pad=pad,
        dilate=dilate,
        no_bias=True,
        weight=weight_q
    )
    return conv

if __name__ == "__main__":
    data_shape = (1, 3, 5, 5)
    conv_w_shape = (4, 3, 3, 3)
    # set random seed
    np.random.seed(5)
    data = np.random.uniform(size=data_shape).astype('float32')
    conv_weight = np.random.uniform(size=conv_w_shape).astype('float32')
    data_names = ['data']
    mx_data_shape = [('data', data_shape)]
    mx_data_batch = mx.io.DataBatch(data=[mx.nd.array(data)])

    data_var = mx.symbol.Variable('data')
    conv = quant_conv_test('test', data_var, 4, is_weight_perchannel=True, w_init_data=conv_weight.tolist())
    flat = mx.symbol.Flatten(data=conv)
    sym = mx.symbol.SoftmaxOutput(data=flat, name='softmax')
    
    internal_syms = sym.get_internals()
    print(internal_syms)

    # build module
    mx_mod = mx.mod.Module(symbol=sym, context=mx.gpu(), data_names=data_names)
    mx_mod.bind(for_training=True, data_shapes=mx_data_shape)

    mx_mod.init_params()
    arg_params, aux_params = mx_mod.get_params()
    init_aux_params = copy.deepcopy(aux_params)
    init_arg_params = copy.deepcopy(arg_params)
    mx_mod.init_optimizer()

    mx_mod.forward(mx_data_batch)
    mx_mod.backward()
    mx_mod.update()
    # mx_mod.forward(mx.io.DataBatch(data=[mx.nd.array(data) + 10]))
    # mx_mod.backward()
    # mx_mod.update()
    mx.nd.waitall()

    new_arg_params, new_aux_params = mx_mod.get_params()
    weight_max = np.array([np.max(channel_data) for channel_data in np.split(conv_weight, conv_weight.shape[0], axis=0)])
    data_max = np.array(np.max(data))

    # np.testing.assert_allclose(data_max, new_aux_params["test_data_quant_minmax"].asnumpy())
    # np.testing.assert_allclose(weight_max, new_aux_params["test_weight_quant_minmax"].asnumpy())

    print('np.max(data):{}, init aux:{}, updated aux:{}'.format(data_max,
        init_aux_params["test_data_quant_minmax"], new_aux_params["test_data_quant_minmax"]))
    print('np.max(weight):{}, init aux:{}, updated aux:{}'.format(weight_max,
        init_aux_params["test_weight_quant_minmax"], new_aux_params["test_weight_quant_minmax"]))

