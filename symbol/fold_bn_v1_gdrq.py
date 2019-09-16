import mxnet as mx
import numpy as np
import copy

class GDRQ_Fold_BN(mx.operator.CustomOp):
    def __init__(self, quant_mode, is_weight_perchannel, delay_quant, ema_decay,
                 name, num_filter, num_group, kernel, stride, pad, dilate, no_bias, 
                 eps, momentum, fix_gamma, quantize_flag):
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
        assert self.no_bias == True, "fold bn don't support bias mode in conv or deconv"
        # bn params
        self.eps = eps
        self.momentum = momentum
        self.fix_gamma = fix_gamma
        # for debug
        self.quantize_flag = quantize_flag

    def forward(self, is_train, req, in_data, out_data, aux):
        assert len(in_data) == 7, "fold bn require seven inputs: data, weight, bn_output, bn_gamma, bn_beta, bn_mean, bn_var"
        data = in_data[0]
        weight = in_data[1]
        bn_gamma = in_data[3]
        bn_beta = in_data[4]
        bn_mean = in_data[5]
        bn_var = in_data[6]

        # print("bn_gamma:{}\n bn_beta:{}\n bn_mean:{}\n bn_var:{}".format(bn_gamma, bn_beta, bn_mean, bn_var))

        if is_train and self.delay_quant > 0:
            # assign bn output to output
            self.assign(out_data[0], req[0], in_data[2])
            self.delay_quant -= 1
            return
        # check_fold_bn_consistence(bn_output=in_data[2], data=in_data[0], weight=weight,
        #                           bn_gamma=bn_gamma, bn_beta=bn_beta, bn_mean=bn_mean, bn_var=bn_var,
        #                           num_filter=self.num_filter, kernel=self.kernel, num_group=self.num_group, 
        #                           stride=self.stride, pad=self.pad, dilate=self.dilate, no_bias=self.no_bias)

        if self.quantize_flag:
            # quantize input
            if is_train:
                data_abs = mx.nd.abs(in_data[0])
                # thresholds = mx.nd.max(data_abs)
                thresholds = 2 * mx.nd.mean(data_abs)
                # udpate acativation thresholds
                if self.init:
                    aux[0][:] = thresholds
                    self.init = False
                else:
                    aux[0][:] = aux[0] * self.ema_decay + thresholds * (1 - self.ema_decay)
            quant_unit = aux[0] / self.QUANT_LEVEL
            # clip data
            data = mx.nd.clip(data, -thresholds.asnumpy()[0], thresholds.asnumpy()[0])
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
                # thresholds = mx.nd.max(weight_abs, axis=reduce_axis[1:])
                thresholds = 2 * mx.nd.mean(weight_abs, axis=reduce_axis[1:])
                # the clip arguments of min/max only support scalar
                for i in range(weight.shape[0]):
                    weight[i,:] = mx.nd.clip(weight[i], -thresholds.asnumpy()[i], thresholds.asnumpy()[i])
                quant_unit = thresholds / self.QUANT_LEVEL
                quant_unit = quant_unit.reshape(w_target_shape).broadcast_like(weight)
            else:
                # thresholds = mx.nd.max(weight_abs)
                thresholds = 2 * mx.nd.mean(weight_abs)
                # clip weight
                weight = mx.nd.clip(weight, -thresholds.asnumpy()[0], thresholds.asnumpy()[0])
                quant_unit = thresholds / self.QUANT_LEVEL
            if is_train:
                aux[1][:] = thresholds
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
        # print("flod bn err:{}".format(np.linalg.norm(fold_bn_result.asnumpy() - in_data[2].asnumpy())))

        self.assign(out_data[0], req[0], conv + bias)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        # assign out grad to the output of bn and assign others to 0
        for i in range(len(in_data)):
            in_grad[i][:] = 0
            # self.assign(in_grad[i], req[i], 0)
            # print("{}:{},{}, sum:{}".format(i, in_data[i].shape, in_grad[i].shape, mx.nd.sum(in_grad[i])))
        # in fold bn, the grad must pass strightforwardly. can't do clip operator
        self.assign(in_grad[2], req[2], out_grad[0])

@mx.operator.register("GDRQ_Fold_BN")
class GDRQ_FoldBNProp(mx.operator.CustomOpProp):
    def __init__(self, quant_mode, is_weight_perchannel=False, delay_quant=0, ema_decay=0.99,
                 name='fold_bn', num_filter=None, num_group=None, kernel=(3,3), stride=(1,1), 
                 pad=(0,0), dilate=(1,1), no_bias=True, eps=1e-5, momentum=0.9, 
                 fix_gamma=False, quantize_flag="True"):
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
        # for debug
        self.quantize_flag = eval(quantize_flag)


        super(GDRQ_FoldBNProp, self).__init__(True)
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
        return GDRQ_Fold_BN(self.quant_mode, self.is_weight_perchannel, self.delay_quant, self.ema_decay,
                       self.name, self.num_filter, self.num_group, self.kernel, self.stride, self.pad, 
                       self.dilate, self.no_bias, self.eps, self.momentum, self.fix_gamma, 
                       self.quantize_flag)


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

    print("bn err:{}".format(np.linalg.norm(fold_bn_result.asnumpy() - bn_output.asnumpy())))
    # raise NotImplementedError

def GDRQ_fold_bn(name, data, 
            # quant params
            quant_mod='minmax', is_weight_perchannel=False, delay_quant=0, ema_decay=0.99,
            # conv params
            num_filter=None, kernel=None, stride=None, pad=(0,0), no_bias=True, dilate=(1,1), num_group=1,
            w_lr_mult=None, w_wd_mult=None, w_init=None,
            # bn params
            eps=1e-5, momentum=0.9, fix_gamma=False, use_global_stats=False,
            gamma_lr_mult=None, gamma_wd_mult=None, gamma_init=None,
            beta_lr_mult=None, beta_wd_mult=None, beta_init=None,
            #for debug
            quantize_flag=True,
            dict_shapes=None):
    if w_init is not None:
        assert isinstance(w_init, mx.init.Initializer)
    if gamma_init is not None:
        assert isinstance(gamma_init, mx.init.Initializer)
    if beta_init is not None:
        assert isinstance(beta_init, mx.init.Initializer)

    if is_weight_perchannel:
        assert quant_mod == "minmax", "currenet weight perchannel only support minmax node with weight"
    input_channel = get_sym_output_channel(name, data, dict_shapes=dict_shapes)
    weight = mx.sym.Variable(name=name + "_conv2d_weight", shape=(num_filter, input_channel // num_group, 
                             kernel[0], kernel[1]), dtype="float32",
                             lr_mult=w_lr_mult, wd_mult=w_wd_mult, init=w_init)
    bn_gamma_var = mx.symbol.Variable(name + '_batchnorm_gamma', shape=(input_channel,), dtype="float32",
                                      lr_mult=gamma_lr_mult, wd_mult=gamma_wd_mult, init=gamma_init)
    bn_beta_var = mx.symbol.Variable(name + '_batchnorm_beta', shape=(input_channel,), dtype="float32",
                                     lr_mult=beta_lr_mult, wd_mult=beta_wd_mult, init=beta_init)

    # conv + bn
    conv = mx.sym.Convolution(name=name + "_conv2d", data=data, weight=weight, num_filter=num_filter, kernel=kernel, num_group=num_group, 
                              stride=stride, pad=pad, no_bias=no_bias, dilate=dilate)
    bn_output_var, bn_mean_var, bn_var_var = mx.sym.BatchNorm_v1(name=name + "_batchnorm", data=conv, gamma=bn_gamma_var, beta=bn_beta_var, 
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
                            # for debug
                            quantize_flag=quantize_flag,
                            op_type="GDRQ_Fold_BN")
    return fold_bn

def simple_sym(custom_op_name):
    assert custom_op_name in ["GDRQ_Fold_BN"]
    name = "test"
    suffix = ""
    quant_mod="minmax"
    is_weight_perchannel=False
    delay_quant=0
    num_filter=4
    kernel=(3,3)
    stride=(1,1)
    pad=(0,0)
    num_group=1
    use_global_stats=False
    fix_gamma = False
    #for debug
    quantize_flag=True

    data = mx.symbol.Variable(name="data")  # 224
    bn = GDRQ_fold_bn(name=name, data=data,
                # quant params
                quant_mod=quant_mod, is_weight_perchannel=is_weight_perchannel, delay_quant=delay_quant, ema_decay=0.99,
                # conv params
                num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, no_bias=True, 
                dilate=(1,1), num_group=num_group,
                # bn params
                eps=1e-5, momentum=0.9, fix_gamma=fix_gamma,use_global_stats=use_global_stats,
                # for debug
                quantize_flag=quantize_flag,
                dict_shapes={"data":(1,3,4,4)})

    act = mx.sym.Activation(data=bn, act_type='relu', name='%s%s_relu' % (name, suffix))
    flat = mx.symbol.Flatten(data=act)
    sym = mx.symbol.SoftmaxOutput(data=flat, name='softmax')
    return sym

def simple_train(custom_op_name):
    sym = simple_sym(custom_op_name)
    # build module
    data_shape = (1, 3, 5, 5)
    # set random seed
    np.random.seed(5)
    data = np.random.uniform(size=data_shape).astype('float32')
    data_names = ['data']
    mx_data_shape = [('data', data_shape)]
    mx_data_batch = mx.io.DataBatch(data=[mx.nd.array(data)], provide_data=mx_data_shape)

    mx_mod = mx.mod.Module(symbol=sym, context=mx.gpu(), data_names=data_names)
    mx_mod.bind(for_training=True, data_shapes=mx_data_shape)

    # # set arg, aux
    # _, arg_params, aux_params = mx.model.load_checkpoint('test_fold_bn_v1', 1)
    # mx_mod.init_params(arg_params=arg_params, aux_params=aux_params)

    mx_mod.init_params()
    arg_params, aux_params = mx_mod.get_params()
    mx_mod.init_optimizer()


    mx_mod.forward(mx_data_batch)
    mx_mod.backward()
    mx_mod.update()
    outputs = mx_mod.get_outputs()
    mx.nd.waitall()
    # callback = mx.callback.do_checkpoint("test_fold_bn_v1")
    # arg_params, aux_params = mx_mod.get_params()
    # callback(0, sym, arg_params, aux_params)
    print("custom op:{} one iter finished".format(custom_op_name))

def simple_test():
    sym, arg_params, aux_params = mx.model.load_checkpoint('test_fold_bn_v1', 1)
    # sym = simple_sym()
    # build module
    data_shape = (1, 3, 5, 5)
    # set random seed
    np.random.seed(5)
    data = np.random.uniform(size=data_shape).astype('float32')
    data_names = ['data']
    mx_data_shape = [('data', data_shape)]
    mx_data_batch = mx.io.DataBatch(data=[mx.nd.array(data)], provide_data=mx_data_shape)

    mx_mod = mx.mod.Module(symbol=sym, context=mx.gpu(), data_names=data_names)
    mx_mod.bind(for_training=False, data_shapes=mx_data_shape)

    # set arg, aux
    mx_mod.init_params(arg_params=arg_params, aux_params=aux_params)
    arg_params, aux_params = mx_mod.get_params()
    mx_mod.init_optimizer()


    mx_mod.forward(mx_data_batch)
    outputs = mx_mod.get_outputs()
    mx.nd.waitall()

if __name__ == "__main__":
    # simple_test()
    simple_train("GDRQ_Fold_BN")