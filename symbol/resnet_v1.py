# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

'''
Adapted from https://github.com/tornadomeet/ResNet/blob/master/symbol_resnet.py
(Original author Wei Wu) by Antti-Pekka Hynninen

Implementing the original resnet ILSVRC 2015 winning network from:

Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Deep Residual Learning for Image Recognition"
'''
import mxnet as mx
import numpy as np

def residual_unit(data, num_filter, stride, dim_match, name, bottle_neck=True, bn_mom=0.9, workspace=256, memonger=False):
    if bottle_neck:
        conv1 = mx.sym.Convolution(data=data, num_filter=int(num_filter*0.25), kernel=(1,1), stride=stride, pad=(0,0),
                                   no_bias=True, workspace=workspace, name=name + '_conv1')
        bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1', cudnn_off=True)
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv2 = mx.sym.Convolution(data=act1, num_filter=int(num_filter*0.25), kernel=(3,3), stride=(1,1), pad=(1,1),
                                   no_bias=True, workspace=workspace, name=name + '_conv2')
        bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2', cudnn_off=True)
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv3 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True,
                                   workspace=workspace, name=name + '_conv3')
        bn3 = mx.sym.BatchNorm(data=conv3, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3', cudnn_off=True)

        if dim_match:
            shortcut = data
        else:
            conv1sc = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_conv1sc')
            shortcut = mx.sym.BatchNorm(data=conv1sc, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_sc', cudnn_off=True)
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return mx.sym.Activation(data=bn3 + shortcut, act_type='relu', name=name + '_relu3')
    else:
        conv1 = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(3,3), stride=stride, pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv1')
        bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1', cudnn_off=True)
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv2 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv2')
        bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2', cudnn_off=True)

        if dim_match:
            shortcut = data
        else:
            conv1sc = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_conv1sc')
            shortcut = mx.sym.BatchNorm(data=conv1sc, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_sc', cudnn_off=True)
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return mx.sym.Activation(data=bn2 + shortcut, act_type='relu', name=name + '_relu3')

def resnet_v1(units, num_stage, filter_list, num_classes, data_type, bottle_neck=True,
           bn_mom=0.9, workspace=256, memonger=False, grad_scale=1.0):
    num_unit = len(units)
    assert(num_unit == num_stage)
    data = mx.sym.Variable(name='data')
    if data_type == 'float32':
        data = mx.sym.identity(data=data, name='id')
    else:
        if data_type == 'float16':
            data = mx.sym.Cast(data=data, dtype=np.float16)
    body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(7, 7), stride=(2,2), pad=(3, 3),
                              no_bias=True, name="conv0", workspace=workspace)
    body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0', cudnn_off=True)
    body = mx.sym.Activation(data=body, act_type='relu', name='relu0')
    body = mx.sym.Pooling(data=body, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')

    for i in range(num_stage):
        body = residual_unit(body, filter_list[i+1], (1 if i==0 else 2, 1 if i==0 else 2), False,
                             name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, workspace=workspace,
                             memonger=memonger)
        for j in range(units[i]-1):
            body = residual_unit(body, filter_list[i+1], (1,1), True, name='stage%d_unit%d' % (i + 1, j + 2),
                                 bottle_neck=bottle_neck, workspace=workspace, memonger=memonger)
    # Although kernel is not used here when global_pool=True, we should put one
    pool1 = mx.sym.Pooling(data=body, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')
    flat = mx.sym.Flatten(data=pool1)
    fc1 = mx.sym.FullyConnected(data=flat, num_hidden=num_classes, name='fc1')
    if data_type == 'float16':
        fc1 = mx.sym.Cast(data=fc1, dtype=np.float32)
        cls = mx.symbol.SoftmaxOutput(data=fc1, name='softmax', grad_scale=grad_scale)
    else:
        cls = mx.symbol.SoftmaxOutput(data=fc1, name='softmax')
    return cls
