import mxnet as mx

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
                channels = data.shape[0]
                target_shape = (channels,) + (1,) * len(data.shape[1:])
                splited_data = mx.nd.split(data, num_outputs=channels, axis=0)
                maxs = mx.nd.array([mx.nd.max(channel_data).asscalar() for channel_data in splited_data])
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

def quant_fc(name, data, num_hidden, quant_mod='minmax', delay_quant=0, is_weight_perchannel=False):
    if is_weight_perchannel:
        assert quant_mod == "minmax", "currenet weight perchannel only support minmax node with weight"
    input_channel = get_sym_output_channel(name, data)
    fc_weight = mx.sym.Variable(name=name +"_weight", shape=(num_hidden, input_channel), dtype="float32")
    fc_q = mx.sym.Custom(fc_weight, name= name + "_weight_quant", is_weight=True, ema_decay=0.99, 
                         delay_quant=delay_quant, quant_mode = quant_mod, is_weight_perchannel=is_weight_perchannel,
                         op_type="Quantization_int8_V2")
    fc_data_q = mx.sym.Custom(data=data, name= name + "_data_quant", is_weight=False, ema_decay=0.99, 
                              delay_quant=delay_quant, quant_mode = quant_mod, is_weight_perchannel=False,
                              op_type="Quantization_int8_V2")
    fc = mx.symbol.FullyConnected(data=fc_data_q, num_hidden=num_hidden, name='fc', weight=fc_q)
    return fc