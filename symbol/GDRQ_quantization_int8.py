import mxnet as mx
import numpy as np
import copy

class GDRQ_Quantization_int8(mx.operator.CustomOp):
    def __init__(self, quant_mode, is_weight, is_weight_perchannel, delay_quant, ema_decay):
        self.quant_mode = quant_mode
        self.is_weight = is_weight
        self.is_weight_perchannel = is_weight_perchannel
        self.delay_quant = delay_quant
        self.ema_decay = ema_decay
        self.QUANT_LEVEL = 127
        self.init = True
        # assert self.is_weight_perchannel == False, "currently GDRQ only support per tensor quantization"
    def forward(self, is_train, req, in_data, out_data, aux):
        if is_train and self.delay_quant > 0:
            self.assign(out_data[0], req[0], in_data[0])
            self.delay_quant -= 1
            return
        if self.is_weight:
            if self.is_weight_perchannel:
                target_shape = (in_data[0].shape[0],) + (1,) * len(in_data[0].shape[1:])
                # save weight thresholds
                if is_train:
                    reduce_axis = tuple([i for i in range(len(in_data[0].shape))])
                    thresholds = 2 * mx.nd.mean(mx.nd.abs(in_data[0]), axis=reduce_axis[1:])
                    aux[0][:] = thresholds
                quant_unit = aux[0] / self.QUANT_LEVEL
                quant_unit = quant_unit.reshape(target_shape)
                # the arguments of min/max only support scalar
                for i in range(in_data[0].shape[0]):
                    out_data[0][i,:] = mx.nd.clip(in_data[0][i], -aux[0].asnumpy()[i], aux[0].asnumpy()[i])
                out_data[0][:] = mx.nd.round(out_data[0] /quant_unit)  * quant_unit
            else:
                # save weight thresholds
                if is_train:
                    thresholds = 2 * mx.nd.mean(mx.nd.abs(in_data[0]))
                    aux[0][:] = thresholds
                quant_unit = aux[0] / self.QUANT_LEVEL
                out_data[0][:] = mx.nd.clip(in_data[0], 
                                            - aux[0].asnumpy()[0], 
                                            aux[0].asnumpy()[0])
                out_data[0][:] = mx.nd.round(out_data[0] / quant_unit) * quant_unit
        else:
            if is_train:
                thresholds = 2 * mx.nd.mean(mx.nd.abs(in_data[0]))
                # udpate activation thresholds
                if self.init:
                    aux[0][:] = thresholds
                    self.init = False
                else:
                    aux[0][:] = aux[0] * self.ema_decay + thresholds * (1 - self.ema_decay)
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
        

@mx.operator.register("GDRQ_Quantization_int8")
class GDRQQuantizationInt8Prop(mx.operator.CustomOpProp):
    def __init__(self, quant_mode, is_weight, is_weight_perchannel=False, delay_quant=0, ema_decay=0.99):
        self.quant_mode = str(quant_mode)
        self.delay_quant = int(delay_quant)
        self.ema_decay = float(ema_decay)
        self.is_weight = eval(is_weight)
        self.is_weight_perchannel = eval(is_weight_perchannel)
        super(GDRQQuantizationInt8Prop, self).__init__(True)
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
        return GDRQ_Quantization_int8(self.quant_mode, self.is_weight,
                                 self.is_weight_perchannel,
                                 self.delay_quant, self.ema_decay)

def get_sym_output_channel(name, sym, dict_shapes=None):
    assert dict_shapes is not None, "please setting dict_shapes for infer shape"
    arguments = sym.list_arguments()
    infer_dict = {}
    for k,v in dict_shapes.items():
        if k in arguments:
            infer_dict[k]=v
    _, out_shapes, _ = sym.infer_shape(**infer_dict)
    assert len(out_shapes) == 1, 'the output of sym is not equal to 1'
    # print('sym:{}:{}'.format(name, sym))
    return out_shapes[0][1]


def GDRQ_quant_conv(name, data, num_filter, kernel, stride, pad=(0,0), no_bias=False, dilate=(1,1), num_group=1,
               lr_mult=None, wd_mult=None, init=None,
               quant_mod='minmax', delay_quant=0, is_weight_perchannel=False, dict_shapes=None):
    if init is not None:
        assert isinstance(init, mx.init.Initializer)
    if is_weight_perchannel:
        assert quant_mod == "minmax", "currenet weight perchannel only support minmax node with weight"

    input_channel = get_sym_output_channel(name, data, dict_shapes=dict_shapes)
    weight = mx.sym.Variable(name=name + "_weight", shape=(num_filter, input_channel // num_group, kernel[0], kernel[1]), 
                             dtype="float32", lr_mult=lr_mult, wd_mult=wd_mult, init=init)
    weight_q = mx.sym.Custom(data=weight, name = name + "_weight", quant_mode=quant_mod, is_weight=True,
                             is_weight_perchannel = is_weight_perchannel, ema_decay=0.99, delay_quant=delay_quant, 
                             op_type="GDRQ_Quantization_int8")
    data_q = mx.sym.Custom(data=data, name = name + "_data", quant_mode=quant_mod, is_weight=False,
                             is_weight_perchannel = False, ema_decay=0.99, delay_quant=delay_quant, 
                             op_type="GDRQ_Quantization_int8")
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

def GDRQ_quant_fc(name, data, num_hidden, flatten=True, no_bias=False,
             lr_mult=None, wd_mult=None, init=None,
             quant_mod='minmax', delay_quant=0, is_weight_perchannel=False, dict_shapes=None):
    if init is not None:
        assert isinstance(init, mx.init.Initializer)
    if is_weight_perchannel:
        assert quant_mod == "minmax", "currenet weight perchannel only support minmax node with weight"
    input_channel = get_sym_output_channel(name, data, dict_shapes=dict_shapes)
    fc_weight = mx.sym.Variable(name=name +"_weight", shape=(num_hidden, input_channel), dtype="float32",
                                lr_mult=lr_mult, wd_mult=wd_mult, init=init)
    fc_q = mx.sym.Custom(fc_weight, name= name + "_weight", is_weight=True, ema_decay=0.99, 
                         delay_quant=delay_quant, quant_mode = quant_mod, is_weight_perchannel=is_weight_perchannel,
                         op_type="GDRQ_Quantization_int8")
    fc_data_q = mx.sym.Custom(data=data, name= name + "_data", is_weight=False, ema_decay=0.99, 
                              delay_quant=delay_quant, quant_mode = quant_mod, is_weight_perchannel=False,
                              op_type="GDRQ_Quantization_int8")
    fc = mx.symbol.FullyConnected(data=fc_data_q, num_hidden=num_hidden, name= name +'fc', weight=fc_q,
                                  flatten=flatten, no_bias=no_bias)
    return fc

def GDRQ_quant_deconv(name, data, kernel, stride, pad, num_filter, no_bias=True, cudnn_tune='fastest', 
                 lr_mult=None, wd_mult=None, init=None,
                 quant_mod='minmax', delay_quant=0, is_weight_perchannel=False, dict_shapes=None):
    if init is not None:
        assert isinstance(init, mx.init.Initializer)
    if is_weight_perchannel:
        assert quant_mod == "minmax", "currenet weight perchannel only support minmax node with weight"
    input_channel = get_sym_output_channel(name, data, dict_shapes=dict_shapes)
    weight = mx.sym.Variable(name=name + "_weight", shape=(input_channel, num_filter,
                                    kernel[0], kernel[1]), dtype="float32",
                                    lr_mult=lr_mult, wd_mult=wd_mult, init=init)
    weight_q = mx.sym.Custom(data=weight, name = name + "_weight", quant_mode=quant_mod, is_weight=True,
                             is_weight_perchannel = is_weight_perchannel, ema_decay=0.99, delay_quant=delay_quant, 
                             op_type="GDRQ_Quantization_int8")
    data_q = mx.sym.Custom(data=data, name = name + "_data", quant_mode=quant_mod, is_weight=False,
                             is_weight_perchannel = False, ema_decay=0.99, delay_quant=delay_quant, 
                             op_type="GDRQ_Quantization_int8")
    deconv = mx.symbol.Deconvolution(name=name, data=data_q, kernel=kernel,stride=stride,pad=pad,
                                   no_bias=no_bias ,num_filter=num_filter,cudnn_tune=cudnn_tune,
                                   weight=weight_q)
    return deconv

def GDRQ_quant_data(name, data, quant_mod='minmax', delay_quant=0, ema_decay=0.99):
    return mx.sym.Custom(data=data, name = name + "_data", quant_mode=quant_mod, is_weight=False,
                             ema_decay=ema_decay, delay_quant=delay_quant, 
                             op_type="GDRQ_Quantization_int8")

def GDRQ_quant_add(name, lhs_data, rhs_data, quant_mod='minmax', delay_quant=0, ema_decay=0.99):
    lhs_data_q = mx.sym.Custom(data=lhs_data, name = name + "add_lhs_data", quant_mode=quant_mod, is_weight=False,
                             ema_decay=ema_decay, delay_quant=delay_quant, is_weight_perchannel=False,
                             op_type="GDRQ_Quantization_int8")
    rhs_data_q = mx.sym.Custom(data=rhs_data, name = name + "add_rhs_data", quant_mode=quant_mod, is_weight=False,
                             ema_decay=ema_decay, delay_quant=delay_quant, is_weight_perchannel=False,
                             op_type="GDRQ_Quantization_int8")
    return mx.symbol.ElementWiseSum(lhs_data_q, rhs_data_q, name =  name + "_plus")

def GDRQ_quant_concat(name, inputs, dim=1, quant_mod='minmax', delay_quant=0, ema_decay=0.99):
    assert isinstance(inputs, list), "the input fo quantize concat must be a list"
    inputs_q = [None] * len(inputs)
    for i in range(len(inputs)):
        inputs_q[i] = mx.sym.Custom(data=inputs[i], name = name + "concat_{}_data".format(i), quant_mode=quant_mod, is_weight=False,
                             ema_decay=ema_decay, delay_quant=delay_quant, is_weight_perchannel=False,
                             op_type="GDRQ_Quantization_int8")
    return mx.symbol.concat(*inputs_q, dim=dim,name=name)