import mxnet as mx 
import numpy as np 
from .clip_grad_quantization_int8 import *


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


def clipgrad_quant_conv(name, data, num_filter, kernel, stride, pad=(0,0), no_bias=False, dilate=(1,1), num_group=1,
               lr_mult=None, wd_mult=None, init=None, weight=None, bias=None,
               quant_mode='minmax', delay_quant=0, is_weight_perchannel=False, ema_decay=0.99, dict_shapes=None):
    if init is not None:
        assert isinstance(init, mx.init.Initializer)
    if is_weight_perchannel:
        assert quant_mode == "minmax", "currenet weight perchannel only support minmax node with weight"

    input_channel = get_sym_output_channel(name, data, dict_shapes=dict_shapes)
    if not isinstance(weight, mx.sym.Symbol) or weight is None:
        weight = mx.sym.Variable(name=name + "_weight", shape=(num_filter, input_channel // num_group, kernel[0], kernel[1]), 
                                dtype="float32", lr_mult=lr_mult, wd_mult=wd_mult, init=init)
    weight_q = mx.sym.Custom(data=weight, name = name + "_weight", quant_mode=quant_mode, is_weight=True,
                             is_weight_perchannel = is_weight_perchannel, ema_decay=ema_decay, delay_quant=delay_quant, 
                             op_type="ClipGrad_Quantization_int8")
    data_q = mx.sym.Custom(data=data, name = name + "_data", quant_mode=quant_mode, is_weight=False,
                             is_weight_perchannel = False, ema_decay=ema_decay, delay_quant=delay_quant, 
                             op_type="ClipGrad_Quantization_int8")
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
        weight=weight_q,
        bias=bias
    )
    return conv

def clipgrad_quant_fc(name, data, num_hidden, flatten=True, no_bias=False,
             lr_mult=None, wd_mult=None, init=None, weight=None, bias=None,
             quant_mode='minmax', delay_quant=0, is_weight_perchannel=False, ema_decay=0.99, dict_shapes=None):
    if init is not None:
        assert isinstance(init, mx.init.Initializer)
    if is_weight_perchannel:
        assert quant_mode == "minmax", "currenet weight perchannel only support minmax node with weight"
    input_channel = get_sym_output_channel(name, data, dict_shapes=dict_shapes)
    if not isinstance(weight, mx.sym.Symbol) or weight is None:
        weight = mx.sym.Variable(name=name +"_weight", shape=(num_hidden, input_channel), dtype="float32",
                                    lr_mult=lr_mult, wd_mult=wd_mult, init=init)
    fc_q = mx.sym.Custom(weight, name= name + "_weight", is_weight=True, ema_decay=ema_decay, 
                         delay_quant=delay_quant, quant_mode = quant_mode, is_weight_perchannel=is_weight_perchannel,
                         op_type="ClipGrad_Quantization_int8")
    fc_data_q = mx.sym.Custom(data=data, name= name + "_data", is_weight=False, ema_decay=ema_decay, 
                              delay_quant=delay_quant, quant_mode = quant_mode, is_weight_perchannel=False,
                              op_type="ClipGrad_Quantization_int8")
    fc = mx.symbol.FullyConnected(data=fc_data_q, num_hidden=num_hidden, name= name, weight=fc_q,
                                  flatten=flatten, no_bias=no_bias, bias=bias)
    return fc

def clipgrad_quant_deconv(name, data, kernel, stride, pad, num_filter, no_bias=True, cudnn_tune='fastest', 
                 lr_mult=None, wd_mult=None, init=None, weight=None, bias=None,
                 quant_mode='minmax', delay_quant=0, is_weight_perchannel=False, ema_decay=0.99, dict_shapes=None):
    if init is not None:
        assert isinstance(init, mx.init.Initializer)
    if is_weight_perchannel:
        assert quant_mode == "minmax", "currenet weight perchannel only support minmax node with weight"
    input_channel = get_sym_output_channel(name, data, dict_shapes=dict_shapes)
    if not isinstance(weight, mx.sym.Symbol) or weight is None:
        weight = mx.sym.Variable(name=name + "_weight", shape=(input_channel, num_filter,
                                        kernel[0], kernel[1]), dtype="float32",
                                        lr_mult=lr_mult, wd_mult=wd_mult, init=init)
    weight_q = mx.sym.Custom(data=weight, name = name + "_weight", quant_mode=quant_mode, is_weight=True,
                             is_weight_perchannel = is_weight_perchannel, ema_decay=ema_decay, delay_quant=delay_quant, 
                             op_type="ClipGrad_Quantization_int8")
    data_q = mx.sym.Custom(data=data, name = name + "_data", quant_mode=quant_mode, is_weight=False,
                             is_weight_perchannel = False, ema_decay=ema_decay, delay_quant=delay_quant, 
                             op_type="ClipGrad_Quantization_int8")
    deconv = mx.symbol.Deconvolution(name=name, data=data_q, kernel=kernel,stride=stride,pad=pad,
                                   no_bias=no_bias ,num_filter=num_filter,cudnn_tune=cudnn_tune,
                                   weight=weight_q, bias=bias)
    return deconv

def clipgrad_quant_data(name, data, quant_mode='minmax', delay_quant=0, ema_decay=0.99):
    return mx.sym.Custom(data=data, name = name + "_data", quant_mode=quant_mode, is_weight=False,
                             ema_decay=ema_decay, delay_quant=delay_quant, 
                             op_type="ClipGrad_Quantization_int8")

def clipgrad_quant_add(name, lhs_data, rhs_data, quant_mode='minmax', delay_quant=0, ema_decay=0.99):
    lhs_data_q = mx.sym.Custom(data=lhs_data, name = name + "add_lhs_data", quant_mode=quant_mode, is_weight=False,
                             ema_decay=ema_decay, delay_quant=delay_quant, is_weight_perchannel=False,
                             op_type="ClipGrad_Quantization_int8")
    rhs_data_q = mx.sym.Custom(data=rhs_data, name = name + "add_rhs_data", quant_mode=quant_mode, is_weight=False,
                             ema_decay=ema_decay, delay_quant=delay_quant, is_weight_perchannel=False,
                             op_type="ClipGrad_Quantization_int8")
    return mx.symbol.ElementWiseSum(lhs_data_q, rhs_data_q, name =  name + "_plus")

def clipgrad_quant_concat(name, inputs, dim=1, quant_mode='minmax', delay_quant=0, ema_decay=0.99):
    assert isinstance(inputs, list), "the input fo quantize concat must be a list"
    inputs_q = [None] * len(inputs)
    for i in range(len(inputs)):
        inputs_q[i] = mx.sym.Custom(data=inputs[i], name = name + "concat_{}_data".format(i), quant_mode=quant_mode, is_weight=False,
                             ema_decay=ema_decay, delay_quant=delay_quant, is_weight_perchannel=False,
                             op_type="ClipGrad_Quantization_int8")
    return mx.symbol.concat(*inputs_q, dim=dim,name=name)


def quant_conv_cxx(name, data, num_filter, kernel, stride, pad=(0,0), no_bias=False, dilate=(1,1), num_group=1,
               lr_mult=None, wd_mult=None, init=None, weight=None, bias=None,
               quant_mode='minmax', delay_quant=0, is_weight_perchannel=False, ema_decay=0.99, grad_mode="ste", workspace=512, dict_shapes=None):
    if init is not None:
        assert isinstance(init, mx.init.Initializer)
    if is_weight_perchannel:
        assert quant_mode == "minmax", "currenet weight perchannel only support minmax node with weight"
    # mod_dict={"minmax":0, "power2": 1}
    # quant_mode = mod_dict[quant_mode]
    input_channel = get_sym_output_channel(name, data, dict_shapes=dict_shapes)
    if not isinstance(weight, mx.sym.Symbol) or weight is None:
        weight = mx.sym.Variable(name=name + "_weight", shape=(num_filter, input_channel // num_group, kernel[0], kernel[1]), 
                                dtype="float32", lr_mult=lr_mult, wd_mult=wd_mult, init=init)
    weight_q = mx.sym.contrib.Quantization_int8(data=weight, name = name + "_weight", quant_mode=quant_mode, is_weight=True,
                             is_weight_perchannel = is_weight_perchannel, ema_decay=ema_decay, delay_quant=delay_quant, grad_mode=grad_mode, workspace=workspace)
    data_q = mx.sym.contrib.Quantization_int8(data=data, name = name + "_data", quant_mode=quant_mode, is_weight=False,
                             is_weight_perchannel = False, ema_decay=ema_decay, delay_quant=delay_quant, grad_mode=grad_mode, workspace=workspace)
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
        weight=weight_q,
        bias=bias
    )
    return conv

def quant_fc_cxx(name, data, num_hidden, flatten=True, no_bias=False,
             lr_mult=None, wd_mult=None, init=None, weight=None, bias=None,
             quant_mode='minmax', delay_quant=0, is_weight_perchannel=False, ema_decay=0.99, grad_mode="ste", workspace=512, dict_shapes=None):
    if init is not None:
        assert isinstance(init, mx.init.Initializer)
    if is_weight_perchannel:
        assert quant_mode == "minmax", "currenet weight perchannel only support minmax node with weight"
    # mod_dict={"minmax":0, "power2": 1}
    # quant_mode = mod_dict[quant_mode]
    input_channel = get_sym_output_channel(name, data, dict_shapes=dict_shapes)
    if not isinstance(weight, mx.sym.Symbol) or weight is None:
        weight = mx.sym.Variable(name=name +"_weight", shape=(num_hidden, input_channel), dtype="float32",
                                    lr_mult=lr_mult, wd_mult=wd_mult, init=init)
    fc_q = mx.sym.contrib.Quantization_int8(weight, name= name + "_weight", is_weight=True, ema_decay=ema_decay, 
                         delay_quant=delay_quant, quant_mode = quant_mode, is_weight_perchannel=is_weight_perchannel, grad_mode=grad_mode, workspace=workspace)
    fc_data_q = mx.sym.contrib.Quantization_int8(data=data, name= name + "_data", is_weight=False, ema_decay=ema_decay, 
                              delay_quant=delay_quant, quant_mode = quant_mode, is_weight_perchannel=False, grad_mode=grad_mode, workspace=workspace)
    fc = mx.symbol.FullyConnected(data=fc_data_q, num_hidden=num_hidden, name= name, weight=fc_q,
                                  flatten=flatten, no_bias=no_bias, bias=bias)
    return fc

def quant_deconv_cxx(name, data, kernel, stride, pad, num_filter, no_bias=True, cudnn_tune='fastest', 
                 lr_mult=None, wd_mult=None, init=None, weight=None, bias=None,
                 quant_mode='minmax', delay_quant=0, is_weight_perchannel=False, ema_decay=0.99, grad_mode="ste", workspace=512, dict_shapes=None):
    if init is not None:
        assert isinstance(init, mx.init.Initializer)
    if is_weight_perchannel:
        assert quant_mode == "minmax", "currenet weight perchannel only support minmax node with weight"
    input_channel = get_sym_output_channel(name, data, dict_shapes=dict_shapes)
    if not isinstance(weight, mx.sym.Symbol) or weight is None:
        weight = mx.sym.Variable(name=name + "_weight", shape=(input_channel, num_filter,
                                        kernel[0], kernel[1]), dtype="float32",
                                        lr_mult=lr_mult, wd_mult=wd_mult, init=init)
    weight_q = mx.sym.contrib.Quantization_int8(data=weight, name = name + "_weight", quant_mode=quant_mode, is_weight=True,
                             is_weight_perchannel = is_weight_perchannel, ema_decay=ema_decay, delay_quant=delay_quant, 
                             grad_mode=grad_mode, workspace=workspace)
    data_q = mx.sym.contrib.Quantization_int8(data=data, name = name + "_data", quant_mode=quant_mode, is_weight=False,
                             is_weight_perchannel = False, ema_decay=ema_decay, delay_quant=delay_quant,
                             grad_mode=grad_mode, workspace=workspace)
    deconv = mx.symbol.Deconvolution(name=name, data=data_q, kernel=kernel,stride=stride,pad=pad,
                                   no_bias=no_bias ,num_filter=num_filter,cudnn_tune=cudnn_tune,
                                   weight=weight_q, bias=bias)
    return deconv

def quant_add_cxx(name, lhs_data, rhs_data, quant_mode='minmax', delay_quant=0, ema_decay=0.99, grad_mode="ste", workspace=512):
    lhs_data_q = mx.sym.contrib.Quantization_int8(data=lhs_data, name = name + "add_lhs_data", quant_mode=quant_mode, is_weight=False,
                             ema_decay=ema_decay, delay_quant=delay_quant, is_weight_perchannel=False, grad_mode=grad_mode, workspace=workspace)
    rhs_data_q = mx.sym.contrib.Quantization_int8(data=rhs_data, name = name + "add_rhs_data", quant_mode=quant_mode, is_weight=False,
                             ema_decay=ema_decay, delay_quant=delay_quant, is_weight_perchannel=False, grad_mode=grad_mode, workspace=workspace)
    return mx.symbol.ElementWiseSum(lhs_data_q, rhs_data_q, name =  name + "_plus")

def quant_concat_cxx(name, inputs, dim=1, quant_mode='minmax', delay_quant=0, ema_decay=0.99, grad_mode="ste", workspace=512):
    assert isinstance(inputs, list), "the input fo quantize concat must be a list"
    inputs_q = [None] * len(inputs)
    for i in range(len(inputs)):
        inputs_q[i] = mx.sym.contrib.Quantization_int8(data=inputs[i], name = name + "concat_{}_data".format(i), quant_mode=quant_mode, is_weight=False,
                             ema_decay=ema_decay, delay_quant=delay_quant, is_weight_perchannel=False, grad_mode=grad_mode, workspace=workspace)
    return mx.symbol.concat(*inputs_q, dim=dim,name=name)
