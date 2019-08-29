import sys

import mxnet as mx
import numpy as np
from .quant_ops import quant_conv
from .quant_ops import quant_fc

eps = 1e-5

def residual_unit_int8(data, channel, num_filter, stride, dim_match, name, bottle_neck=True,
                       bn_mom=0.9, workspace=512, memonger=False, 
                       quant_mod='minmax', delay_quant=0, is_weight_perchannel=False,
                       use_global_stats=False, fix_gamma=False):
    if bottle_neck:
        bn1 = mx.sym.BatchNorm(data=data, eps=eps, momentum=bn_mom, name=name + '_bn1',
                                fix_gamma=fix_gamma, use_global_stats=use_global_stats)
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = quant_conv(name=name + '_conv1', data=act1, num_filter=int(num_filter * 0.25), kernel=(1, 1), stride=(1, 1),
                           pad=(0, 0), no_bias=True, 
                           quant_mod=quant_mod, delay_quant=delay_quant, is_weight_perchannel=is_weight_perchannel)
        bn2 = mx.sym.BatchNorm(data=conv1, eps=eps, momentum=bn_mom, name=name + '_bn2',
                                fix_gamma=fix_gamma, use_global_stats=use_global_stats)
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = quant_conv(name=name + '_conv2', data=act2, num_filter=int(num_filter * 0.25), kernel=(3, 3), stride=stride,
                           pad=(1, 1), no_bias=True, 
                           quant_mod=quant_mod, delay_quant=delay_quant, is_weight_perchannel=is_weight_perchannel)
        bn3 = mx.sym.BatchNorm(data=conv2, eps=eps, momentum=bn_mom, name=name + '_bn3',
                                fix_gamma=fix_gamma, use_global_stats=use_global_stats)
        act3 = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')
        conv3 = quant_conv(name=name + '_conv3', data=act3, num_filter=num_filter, kernel=(1, 1), 
                           stride=(1, 1), pad=(0, 0), no_bias=True,
                           quant_mod=quant_mod, delay_quant=delay_quant, is_weight_perchannel=is_weight_perchannel)
        if dim_match:
            shortcut = data
        else:
            shortcut = quant_conv(name=name + '_sc', data=act1, num_filter=num_filter, kernel=(1, 1), 
                                  stride=stride, no_bias=True,
                                  quant_mod=quant_mod, delay_quant=delay_quant, is_weight_perchannel=is_weight_perchannel)
        if memonger:
            shortcut._set_attr(mirror_stage='True')

        return conv3 + shortcut
    else:
        bn1 = mx.sym.BatchNorm(data=data, momentum=bn_mom, eps=eps, name=name + '_bn1',
                                fix_gamma=fix_gamma, use_global_stats=use_global_stats)
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = quant_conv(name=name + '_conv1', data=relu1_q, num_filter=num_filter, kernel=(3, 3), 
                           stride=stride, pad=(1, 1), no_bias=True,
                           quant_mod=quant_mod, delay_quant=delay_quant, is_weight_perchannel=is_weight_perchannel)
        bn2 = mx.sym.BatchNorm(data=conv1, momentum=bn_mom, eps=eps, name=name + '_bn2',
                                fix_gamma=fix_gamma, use_global_stats=use_global_stats)
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = quant_conv(name=name + '_conv2', data=relu2_q, num_filter=num_filter, kernel=(3, 3), 
                           stride=(1, 1), pad=(1, 1), no_bias=True,
                           quant_mod=quant_mod, delay_quant=delay_quant, is_weight_perchannel=is_weight_perchannel)
        if dim_match:
            shortcut = data
        else:
            shortcut = quant_conv(name=name + '_sc', data=data_sc_q, num_filter=num_filter, kernel=(1, 1), 
                                  stride=stride, no_bias=True,
                                  quant_mod=quant_mod, delay_quant=delay_quant, is_weight_perchannel=is_weight_perchannel)
        if memonger:
            shortcut._set_attr(mirror_stage='True')

        return conv2 + shortcut

def resnet_int8(units, num_stage, filter_list, num_classes, data_type, bottle_neck=True,
                bn_mom=0.9, workspace=512, memonger=False, grad_scale=1.0, dataset_type=None,
                quant_mod='minmax', delay_quant=0, is_weight_perchannel=False,
                use_global_stats=False, fix_gamma=False):
    num_unit = len(units)
    assert (num_unit == num_stage)
    print('units:{}, num_stage:{}, filter_list:{}, num_classes:{}, data_type:{}, bottle_neck:{},'
          'bn_mom:{}, quant_mod:{}, delay_quant:{}, is_weight_perchannel:{}'.format(
        units, num_stage, filter_list, num_classes, data_type, bottle_neck,
        bn_mom, quant_mod, delay_quant, is_weight_perchannel
    ))
    data = mx.sym.Variable(name='data')
    if data_type == 'float32':
        data = mx.sym.identity(data=data, name='id')
    elif data_type == 'float16':
        data = mx.sym.Cast(data=data, dtype=np.float16)

    data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')

    if dataset_type == 'imagenet':
        body = quant_conv(name="conv0", data=data, num_filter=filter_list[0], kernel=(7, 7), 
                          stride=(2, 2), pad=(3, 3), no_bias=True, 
                          quant_mod=quant_mod, delay_quant=delay_quant, is_weight_perchannel=is_weight_perchannel)
        body = mx.sym.BatchNorm(data=body, eps=eps, momentum=bn_mom, name='bn0',
                                fix_gamma=fix_gamma, use_global_stats=use_global_stats)
        body = mx.sym.Activation(data=body, act_type='relu', name='relu0')
        body = mx.symbol.Pooling(data=body, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max')
    elif dataset_type == 'cifar10':
        body = quant_conv(name="conv0", data=data, num_filter=filter_list[0], kernel=(3, 3), 
                          stride=(1, 1), pad=(1, 1), no_bias=True,
                          quant_mod=quant_mod, delay_quant=delay_quant, is_weight_perchannel=is_weight_perchannel)
    else:
        raise ValueError("resnet only support imagenet or cifar10 dataset, {}".format(dataset_type))

    for i in range(num_stage):
        body = residual_unit_int8(body, filter_list[i], filter_list[i + 1], (1 if i == 0 else 2, 1 if i == 0 else 2), False,
                             name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, workspace=workspace,
                             memonger=memonger, 
                             quant_mod=quant_mod, delay_quant=delay_quant, is_weight_perchannel=is_weight_perchannel,
                             fix_gamma=fix_gamma, use_global_stats=use_global_stats)
        for j in range(int(units[i] - 1)):
            body = residual_unit_int8(body, filter_list[i + 1], filter_list[i + 1], (1, 1), True, 
                                      name='stage%d_unit%d' % (i + 1, j + 2),
                                      bottle_neck=bottle_neck, workspace=workspace, memonger=memonger,
                                      quant_mod=quant_mod, delay_quant=delay_quant, 
                                      is_weight_perchannel=is_weight_perchannel,
                                      fix_gamma=fix_gamma, use_global_stats=use_global_stats)
    bn1 = mx.sym.BatchNorm(data=body, eps=eps, momentum=bn_mom, name='bn1',
                                fix_gamma=fix_gamma, use_global_stats=use_global_stats)
    relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')

    pool1 = mx.symbol.Pooling(data=relu1, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')
    flat = mx.symbol.Flatten(data=pool1)

    # # add quantize for fc
    fc1 = quant_fc(name='fc1', data=flat, num_hidden=num_classes,
                   quant_mod=quant_mod, delay_quant=delay_quant, is_weight_perchannel=is_weight_perchannel)
    if data_type == 'float16':
        fc1 = mx.sym.Cast(data=fc1, dtype=np.float32)
        cls = mx.symbol.SoftmaxOutput(data=fc1, name='softmax', grad_scale=grad_scale)
    else:
        cls = mx.symbol.SoftmaxOutput(data=fc1, name='softmax')
    return cls
