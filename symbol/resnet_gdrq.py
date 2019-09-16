import sys
import config

import mxnet as mx
import numpy as np
from .GDRQ_quantization_int8 import GDRQ_quant_conv, GDRQ_quant_fc
# from .clip_grad_quantization_int8 import clipgrad_quant_conv as GDRQ_quant_conv
# from .clip_grad_quantization_int8 import clipgrad_quant_fc as GDRQ_quant_fc

eps = 1e-5


def residual_unit(data, num_filter, stride, dim_match, name, bottle_neck=True,
                  bn_mom=0.9, workspace=512, memonger=False,
                  quant_mod='minmax', delay_quant=0, is_weight_perchannel=False, dict_shapes={'data':(1, 3, 224, 224)}):
    if bottle_neck:
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=eps, momentum=bn_mom, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        # conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter * 0.25), kernel=(1, 1), stride=(1, 1),
        #                            pad=(0, 0),
        #                            no_bias=True, workspace=workspace, name=name + '_conv1')
        conv1 = GDRQ_quant_conv(data=act1, num_filter=int(num_filter * 0.25), kernel=(1, 1), stride=(1, 1),
                                   pad=(0, 0),
                                   no_bias=True, name=name + '_conv1',
                                   quant_mod=quant_mod, delay_quant=delay_quant, 
                                   is_weight_perchannel=is_weight_perchannel, dict_shapes=dict_shapes)
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=eps, momentum=bn_mom, name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        # conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter * 0.25), kernel=(3, 3), stride=stride,
        #                            pad=(1, 1),
        #                            no_bias=True, workspace=workspace, name=name + '_conv2')
        conv2 = GDRQ_quant_conv(data=act2, num_filter=int(num_filter * 0.25), kernel=(3, 3), stride=stride,
                                   pad=(1, 1),
                                   no_bias=True, name=name + '_conv2',
                                   quant_mod=quant_mod, delay_quant=delay_quant, 
                                   is_weight_perchannel=is_weight_perchannel, dict_shapes=dict_shapes)
        bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=eps, momentum=bn_mom, name=name + '_bn3')
        act3 = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')
        # conv3 = mx.sym.Convolution(data=act3, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
        #                            no_bias=True,
        #                            workspace=workspace, name=name + '_conv3')
        conv3 = GDRQ_quant_conv(data=act3, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                                   no_bias=True,
                                   name=name + '_conv3',
                                   quant_mod=quant_mod, delay_quant=delay_quant, 
                                   is_weight_perchannel=is_weight_perchannel, dict_shapes=dict_shapes)
        if dim_match:
            shortcut = data
        else:
            # shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1, 1), stride=stride, no_bias=True,
            #                               workspace=workspace, name=name + '_sc')
            shortcut = GDRQ_quant_conv(data=act1, num_filter=num_filter, kernel=(1, 1), stride=stride, no_bias=True,
                                        name=name + '_sc',
                                   quant_mod=quant_mod, delay_quant=delay_quant, 
                                   is_weight_perchannel=is_weight_perchannel, dict_shapes=dict_shapes)
        if memonger:
            shortcut._set_attr(mirror_stage='True')

        return conv3 + shortcut
    else:
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, momentum=bn_mom, eps=eps, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        # conv1 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3, 3), stride=stride, pad=(1, 1),
        #                            no_bias=True, workspace=workspace, name=name + '_conv1')
        conv1 = GDRQ_quant_conv(data=act1, num_filter=num_filter, kernel=(3, 3), stride=stride, pad=(1, 1),
                                   no_bias=True, name=name + '_conv1',
                                   quant_mod=quant_mod, delay_quant=delay_quant, 
                                   is_weight_perchannel=is_weight_perchannel, dict_shapes=dict_shapes)
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=eps, name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        # conv2 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
        #                            no_bias=True, workspace=workspace, name=name + '_conv2')
        conv2 = GDRQ_quant_conv(data=act2, num_filter=num_filter, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                                   no_bias=True, name=name + '_conv2',
                                   quant_mod=quant_mod, delay_quant=delay_quant, 
                                   is_weight_perchannel=is_weight_perchannel, dict_shapes=dict_shapes)
        if dim_match:
            shortcut = data
        else:
            # shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1, 1), stride=stride, no_bias=True,
            #                               workspace=workspace, name=name + '_sc')
            shortcut = GDRQ_quant_conv(data=act1, num_filter=num_filter, kernel=(1, 1), stride=stride, no_bias=True,
                                       name=name + '_sc',
                                   quant_mod=quant_mod, delay_quant=delay_quant, 
                                   is_weight_perchannel=is_weight_perchannel, dict_shapes=dict_shapes)
        if memonger:
            shortcut._set_attr(mirror_stage='True')

        return conv2 + shortcut


def resnet_gdrq(units, num_stage, filter_list, num_classes, data_type, bottle_neck=True,
           bn_mom=0.9, workspace=512, memonger=False, grad_scale=1.0, dataset_type=None,
                  quant_mod='minmax', delay_quant=0, is_weight_perchannel=False, dict_shapes={'data':(1, 3, 224, 224)}):
    num_unit = len(units)
    assert (num_unit == num_stage)

    data = mx.sym.Variable(name='data')
    if data_type == 'float32':
        data = mx.sym.identity(data=data, name='id')
    elif data_type == 'float16':
        data = mx.sym.Cast(data=data, dtype=np.float16)

    data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')

    if dataset_type == 'imagenet':
        # body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(7, 7), stride=(2, 2), pad=(3, 3),
        #                           no_bias=True, name="conv0", workspace=workspace)
        body = GDRQ_quant_conv(data=data, num_filter=filter_list[0], kernel=(7, 7), stride=(2, 2), pad=(3, 3),
                                  no_bias=True, name="conv0", 
                                   quant_mod=quant_mod, delay_quant=delay_quant, 
                                   is_weight_perchannel=is_weight_perchannel, dict_shapes=dict_shapes)
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=eps, momentum=bn_mom, name='bn0')
        body = mx.sym.Activation(data=body, act_type='relu', name='relu0')
        body = mx.symbol.Pooling(data=body, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max')
    elif dataset_type in ['cifar10', 'cifar100']:
        # body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(3, 3), stride=(1, 1), pad=(1, 1),
        #                           no_bias=True, name="conv0", workspace=workspace)
        body = GDRQ_quant_conv(data=data, num_filter=filter_list[0], kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                                  no_bias=True, name="conv0", 
                                   quant_mod=quant_mod, delay_quant=delay_quant, 
                                   is_weight_perchannel=is_weight_perchannel, dict_shapes=dict_shapes)
                                
    else:
        raise ValueError("resnet only support imagenet or cifar10 dataset")

    for i in range(num_stage):
        body = residual_unit(body, filter_list[i + 1], (1 if i == 0 else 2, 1 if i == 0 else 2), False,
                             name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, workspace=workspace,
                             memonger=memonger, 
                                   quant_mod=quant_mod, delay_quant=delay_quant, 
                                   is_weight_perchannel=is_weight_perchannel, dict_shapes=dict_shapes)
        for j in range(units[i] - 1):
            body = residual_unit(body, filter_list[i + 1], (1, 1), True, name='stage%d_unit%d' % (i + 1, j + 2),
                                 bottle_neck=bottle_neck, workspace=workspace, memonger=memonger, 
                                   quant_mod=quant_mod, delay_quant=delay_quant, 
                                   is_weight_perchannel=is_weight_perchannel, dict_shapes=dict_shapes)
    bn1 = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=eps, momentum=bn_mom, name='bn1')
    relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')
    pool1 = mx.symbol.Pooling(data=relu1, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')
    flat = mx.symbol.Flatten(data=pool1)
    # fc1 = mx.symbol.FullyConnected(data=flat, num_hidden=num_classes, name='fc1')
    fc1 = GDRQ_quant_fc(data=flat, num_hidden=num_classes, name='fc1', 
                                   quant_mod=quant_mod, delay_quant=delay_quant, 
                                   is_weight_perchannel=is_weight_perchannel, dict_shapes=dict_shapes)
    if data_type == 'float16':
        fc1 = mx.sym.Cast(data=fc1, dtype=np.float32)
        cls = mx.symbol.SoftmaxOutput(data=fc1, name='softmax', grad_scale=grad_scale)
    else:
        cls = mx.symbol.SoftmaxOutput(data=fc1, name='softmax')
    return cls


def resnet_gdrq_cifar(units, num_stage, filter_list, num_classes, data_type, bottle_neck=True,
           bn_mom=0.9, workspace=512, memonger=False, grad_scale=1.0,
                  quant_mod='minmax', delay_quant=0, is_weight_perchannel=False, dict_shapes={'data':(1, 3, 224, 224)}):
    num_unit = len(units)
    assert (num_unit == num_stage)

    data = mx.sym.Variable(name='data')
    if data_type == 'float32':
        data = mx.sym.identity(data=data, name='id')
    elif data_type == 'float16':
        data = mx.sym.Cast(data=data, dtype=np.float16)

    # body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(3, 3), stride=(1, 1), pad=(1, 1),
    #                           no_bias=True, name="conv0", workspace=workspace)
    body = GDRQ_quant_conv(data=data, num_filter=filter_list[0], kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                              no_bias=True, name="conv0", 
                                   quant_mod=quant_mod, delay_quant=delay_quant, 
                                   is_weight_perchannel=is_weight_perchannel, dict_shapes=dict_shapes)

    for i in range(num_stage):
        body = residual_unit(body, filter_list[i + 1], (1 if i == 0 else 2, 1 if i == 0 else 2), False,
                             name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, workspace=workspace,
                             memonger=memonger, 
                                   quant_mod=quant_mod, delay_quant=delay_quant, 
                                   is_weight_perchannel=is_weight_perchannel, dict_shapes=dict_shapes)
        for j in range(units[i] - 1):
            body = residual_unit(body, filter_list[i + 1], (1, 1), True, name='stage%d_unit%d' % (i + 1, j + 2),
                                 bottle_neck=bottle_neck, workspace=workspace, memonger=memonger, 
                                   quant_mod=quant_mod, delay_quant=delay_quant, 
                                   is_weight_perchannel=is_weight_perchannel, dict_shapes=dict_shapes)
    bn1 = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=eps, momentum=bn_mom, name='bn1')
    relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')
    pool1 = mx.symbol.Pooling(data=relu1, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')
    flat = mx.symbol.Flatten(data=pool1)
    # fc1 = mx.symbol.FullyConnected(data=flat, num_hidden=num_classes, name='fc1')
    fc1 = mx.symbol.FullyConnected(data=flat, num_hidden=num_classes, name='fc1', 
                                   quant_mod=quant_mod, delay_quant=delay_quant, 
                                   is_weight_perchannel=is_weight_perchannel, dict_shapes=dict_shapes)
    if data_type == 'float16':
        fc1 = mx.sym.Cast(data=fc1, dtype=np.float32)
        cls = mx.symbol.SoftmaxOutput(data=fc1, name='softmax', grad_scale=grad_scale)
    else:
        cls = mx.symbol.SoftmaxOutput(data=fc1, name='softmax')
    return cls