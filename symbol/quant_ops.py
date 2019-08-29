import mxnet as mx

def get_sym_output_channel(name, sym, data_shape=(1, 3, 224, 224)):
    _, out_shapes, _ = sym.infer_shape(data=data_shape)
    assert len(out_shapes) == 1, 'the output of sym is not equal to 1'
    # print('sym:{}:{}'.format(name, sym))
    return out_shapes[0][1]

def quant_conv(name, data, num_filter, kernel, stride, pad=(0,0), no_bias=True, dilate=(1,1), num_group=1,
               quant_mod='minmax', delay_quant=0, is_weight_perchannel=False):
    if is_weight_perchannel:
        assert quant_mod == "minmax", "currenet weight perchannel only support minmax node with weight"

    input_channel = get_sym_output_channel(name, data)
    weight = mx.sym.Variable(name=name + "_weight", shape=(num_filter, input_channel // num_group, kernel[0], kernel[1]))

    weight_q = mx.sym.Quantization_int8(data = weight, name=name + '_weight_quant', 
                                        is_weight=True, ema_decay=0.99, delay_quant=delay_quant,
                                        is_train=True, quant_mod = quant_mod, is_weight_perchannel=is_weight_perchannel)
    data_q = mx.sym.Quantization_int8(data=data, name=name + '_data_quant',
                                      is_weight=False, ema_decay=0.99, delay_quant=delay_quant,
                                      is_train=True, quant_mod = quant_mod)
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

def quant_fc(name, data, num_hidden, quant_mod='minmax', delay_quant=0, is_weight_perchannel=False):
    if is_weight_perchannel:
        assert quant_mod == "minmax", "currenet weight perchannel only support minmax node with weight"
    input_channel = get_sym_output_channel(name, data)
    fc_weight = mx.sym.Variable(name=name +"_weight", shape=(num_hidden, input_channel))
    fc_q = mx.sym.Quantization_int8(fc_weight, name= name + "_weight_quant",
                                    is_weight=True, ema_decay=0.99, delay_quant=delay_quant,
                                    is_train=True, quant_mod = quant_mod, is_weight_perchannel=is_weight_perchannel)
    fc_data_q = mx.sym.Quantization_int8(data=data, name= name + "_data_quant",
                                         is_weight=False, ema_decay=0.99, delay_quant=delay_quant,
                                         is_train=True, quant_mod = quant_mod)
    fc = mx.symbol.FullyConnected(data=fc_data_q, num_hidden=num_hidden, name='fc', weight=fc_q)
    return fc