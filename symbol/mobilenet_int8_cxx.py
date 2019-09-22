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
# -*- coding:utf-8 -*-
'''
mobilenet
Suittable for image with around resolution x resolution, resolution is multiple of 32.

Reference:
MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
https://arxiv.org/abs/1704.04861
'''

__author__ = 'qingzhouzhen'
__date__ = '17/8/5'
__modify__ = 'dwSun'
__modified_date__ = '17/11/30'


import mxnet as mx
# from .quantization_int8_V2 import quant_conv
# from .quantization_int8_V2 import quant_fc

from .int8_api import quant_conv_cxx as quant_conv
from .int8_api import quant_fc_cxx as quant_fc


alpha_values = [0.25, 0.50, 0.75, 1.0]


def Conv(data, num_filter=1, channel=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, name='', suffix='',
         quant_mode='minmax', grad_mode='ste', delay_quant=0, is_weight_perchannel=False, 
         dict_shapes={"data": (1, 3, 224, 224)}):
    conv = quant_conv(name='%s%s_conv2d' % (name, suffix), data=data, num_filter=num_filter, kernel=kernel, num_group=num_group,
                      stride=stride, pad=pad, no_bias=True, dilate=(1,1), quant_mode=quant_mode, grad_mode=grad_mode, 
                      delay_quant=delay_quant, is_weight_perchannel=is_weight_perchannel, dict_shapes=dict_shapes)

    bn = mx.sym.BatchNorm(data=conv, name='%s%s_batchnorm' % (name, suffix), fix_gamma=True)
    act = mx.sym.Activation(data=bn, act_type='relu', name='%s%s_relu' % (name, suffix))
    return act

def mobilenet_int8_cxx(num_classes, alpha=1, resolution=224, 
                   quant_mode='minmax', grad_mode='ste', delay_quant=0, is_weight_perchannel=False,
                   use_global_stats=False, fix_gamma=False, dict_shapes={"data": (1,3,224,224)}, **kwargs):
    assert alpha in alpha_values, 'Invalid alpha=[{0}], must be one of [{1}]'.format(alpha, alpha_values)
    assert resolution % 32 == 0, 'resolution must be multpile of 32'

    base = int(32 * alpha)

    data = mx.symbol.Variable(name="data")  # 224

    depth = base  # 32*alpha
    conv_1 = Conv(data, num_filter=depth, channel=3, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name="conv_1",
                  quant_mode=quant_mode, grad_mode=grad_mode, delay_quant=delay_quant, is_weight_perchannel=is_weight_perchannel, dict_shapes=dict_shapes)  # 224/112

    depth = base  # 32*alpha
    conv_2_dw = Conv(conv_1, num_group=depth, num_filter=depth, channel=depth, kernel=(3, 3), pad=(1, 1), stride=(1, 1),
                     name="conv_2_dw", 
                     quant_mode=quant_mode, grad_mode=grad_mode, delay_quant=delay_quant, is_weight_perchannel=is_weight_perchannel, dict_shapes=dict_shapes)  # 112/112
    conv_2 = Conv(conv_2_dw, num_filter=depth * 2, channel=depth, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_2",
                  quant_mode=quant_mode, grad_mode=grad_mode, delay_quant=delay_quant, is_weight_perchannel=is_weight_perchannel, dict_shapes=dict_shapes)  # 112/112

    depth = base * 2  # 64*alpha
    conv_3_dw = Conv(conv_2, num_group=depth, num_filter=depth, channel=depth, kernel=(3, 3), pad=(1, 1), stride=(2, 2),
                     name="conv_3_dw",
                     quant_mode=quant_mode, grad_mode=grad_mode, delay_quant=delay_quant, is_weight_perchannel=is_weight_perchannel, dict_shapes=dict_shapes)  # 112/56
    conv_3 = Conv(conv_3_dw, num_filter=depth * 2, channel=depth, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_3",
                  quant_mode=quant_mode, grad_mode=grad_mode, delay_quant=delay_quant, is_weight_perchannel=is_weight_perchannel, dict_shapes=dict_shapes)  # 56/56

    depth = base * 4  # 128*alpha
    conv_4_dw = Conv(conv_3, num_group=depth, num_filter=depth, channel=depth, kernel=(3, 3), pad=(1, 1), stride=(1, 1),
                     name="conv_4_dw",
                     quant_mode=quant_mode, grad_mode=grad_mode, delay_quant=delay_quant, is_weight_perchannel=is_weight_perchannel, dict_shapes=dict_shapes)  # 56/56
    conv_4 = Conv(conv_4_dw, num_filter=depth, channel=depth, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_4",
                  quant_mode=quant_mode, grad_mode=grad_mode, delay_quant=delay_quant, is_weight_perchannel=is_weight_perchannel, dict_shapes=dict_shapes)  # 56/56

    depth = base * 4  # 128*alpha
    conv_5_dw = Conv(conv_4, num_group=depth, num_filter=depth, channel=depth, kernel=(3, 3), pad=(1, 1), stride=(2, 2),
                     name="conv_5_dw",
                     quant_mode=quant_mode, grad_mode=grad_mode, delay_quant=delay_quant, is_weight_perchannel=is_weight_perchannel, dict_shapes=dict_shapes)  # 56/28
    conv_5 = Conv(conv_5_dw, num_filter=depth * 2, channel=depth, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_5",
                  quant_mode=quant_mode, grad_mode=grad_mode, delay_quant=delay_quant, is_weight_perchannel=is_weight_perchannel, dict_shapes=dict_shapes)  # 28/28

    depth = base * 8  # 256*alpha
    conv_6_dw = Conv(conv_5, num_group=depth, num_filter=depth, channel=depth, kernel=(3, 3), pad=(1, 1), stride=(1, 1),
                     name="conv_6_dw",
                     quant_mode=quant_mode, grad_mode=grad_mode, delay_quant=delay_quant, is_weight_perchannel=is_weight_perchannel, dict_shapes=dict_shapes)  # 28/28
    conv_6 = Conv(conv_6_dw, num_filter=depth, channel=depth, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_6",
                  quant_mode=quant_mode, grad_mode=grad_mode, delay_quant=delay_quant, is_weight_perchannel=is_weight_perchannel, dict_shapes=dict_shapes)  # 28/28

    depth = base * 8  # 256*alpha
    conv_7_dw = Conv(conv_6, num_group=depth, num_filter=depth, channel=depth, kernel=(3, 3), pad=(1, 1), stride=(2, 2),
                     name="conv_7_dw",
                     quant_mode=quant_mode, grad_mode=grad_mode, delay_quant=delay_quant, is_weight_perchannel=is_weight_perchannel, dict_shapes=dict_shapes)  # 28/14
    conv_7 = Conv(conv_7_dw, num_filter=depth * 2, channel=depth, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_7",
                  quant_mode=quant_mode, grad_mode=grad_mode, delay_quant=delay_quant, is_weight_perchannel=is_weight_perchannel, dict_shapes=dict_shapes)  # 14/14

    depth = base * 16  # 512*alpha
    conv_8_dw = Conv(conv_7, num_group=depth, num_filter=depth, channel=depth, kernel=(3, 3), pad=(1, 1), stride=(1, 1),
                     name="conv_8_dw",
                     quant_mode=quant_mode, grad_mode=grad_mode, delay_quant=delay_quant, is_weight_perchannel=is_weight_perchannel, dict_shapes=dict_shapes)  # 14/14
    conv_8 = Conv(conv_8_dw, num_filter=depth, channel=depth, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_8",
                  quant_mode=quant_mode, grad_mode=grad_mode, delay_quant=delay_quant, is_weight_perchannel=is_weight_perchannel, dict_shapes=dict_shapes)  # 14/14
    conv_9_dw = Conv(conv_8, num_group=depth, num_filter=depth, channel=depth, kernel=(3, 3), pad=(1, 1), stride=(1, 1),
                     name="conv_9_dw",
                     quant_mode=quant_mode, grad_mode=grad_mode, delay_quant=delay_quant, is_weight_perchannel=is_weight_perchannel, dict_shapes=dict_shapes)  # 14/14
    conv_9 = Conv(conv_9_dw, num_filter=depth, channel=depth, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_9",
                  quant_mode=quant_mode, grad_mode=grad_mode, delay_quant=delay_quant, is_weight_perchannel=is_weight_perchannel, dict_shapes=dict_shapes)  # 14/14
    conv_10_dw = Conv(conv_9, num_group=depth, num_filter=depth, channel=depth, kernel=(3, 3), pad=(1, 1), stride=(1, 1),
                      name="conv_10_dw",
                      quant_mode=quant_mode, grad_mode=grad_mode, delay_quant=delay_quant, is_weight_perchannel=is_weight_perchannel, dict_shapes=dict_shapes)  # 14/14
    conv_10 = Conv(conv_10_dw, num_filter=depth, channel=depth, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_10",
                   quant_mode=quant_mode, grad_mode=grad_mode, delay_quant=delay_quant, is_weight_perchannel=is_weight_perchannel, dict_shapes=dict_shapes)  # 14/14
    conv_11_dw = Conv(conv_10, num_group=depth, num_filter=depth, channel=depth, kernel=(3, 3), pad=(1, 1), stride=(1, 1),
                      name="conv_11_dw",
                      quant_mode=quant_mode, grad_mode=grad_mode, delay_quant=delay_quant, is_weight_perchannel=is_weight_perchannel, dict_shapes=dict_shapes)  # 14/14
    conv_11 = Conv(conv_11_dw, num_filter=depth, channel=depth, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_11",
                   quant_mode=quant_mode, grad_mode=grad_mode, delay_quant=delay_quant, is_weight_perchannel=is_weight_perchannel, dict_shapes=dict_shapes)  # 14/14
    conv_12_dw = Conv(conv_11, num_group=depth, num_filter=depth, channel=depth, kernel=(3, 3), pad=(1, 1), stride=(1, 1),
                      name="conv_12_dw",
                      quant_mode=quant_mode, grad_mode=grad_mode, delay_quant=delay_quant, is_weight_perchannel=is_weight_perchannel, dict_shapes=dict_shapes)  # 14/14
    conv_12 = Conv(conv_12_dw, num_filter=depth, channel=depth, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_12",
                   quant_mode=quant_mode, grad_mode=grad_mode, delay_quant=delay_quant, is_weight_perchannel=is_weight_perchannel, dict_shapes=dict_shapes)  # 14/14

    depth = base * 16  # 512*alpha
    conv_13_dw = Conv(conv_12, num_group=depth, num_filter=depth, channel=depth, kernel=(3, 3), pad=(1, 1), stride=(2, 2),
                      name="conv_13_dw",
                      quant_mode=quant_mode, grad_mode=grad_mode, delay_quant=delay_quant, is_weight_perchannel=is_weight_perchannel, dict_shapes=dict_shapes)  # 14/7
    conv_13 = Conv(conv_13_dw, num_filter=depth * 2, channel=depth, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_13",
                   quant_mode=quant_mode, grad_mode=grad_mode, delay_quant=delay_quant, is_weight_perchannel=is_weight_perchannel, dict_shapes=dict_shapes)  # 7/7

    depth = base * 32  # 1024*alpha
    conv_14_dw = Conv(conv_13, num_group=depth, num_filter=depth, channel=depth, kernel=(3, 3), pad=(1, 1), stride=(1, 1),
                      name="conv_14_dw",
                      quant_mode=quant_mode, grad_mode=grad_mode, delay_quant=delay_quant, is_weight_perchannel=is_weight_perchannel, dict_shapes=dict_shapes)  # 7/7
    conv_14 = Conv(conv_14_dw, num_filter=depth, channel=depth, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_14",
                   quant_mode=quant_mode, grad_mode=grad_mode, delay_quant=delay_quant, is_weight_perchannel=is_weight_perchannel, dict_shapes=dict_shapes)  # 7/7

    pool_size = int(resolution / 32)
    pool = mx.sym.Pooling(data=conv_14, kernel=(pool_size, pool_size), stride=(1, 1), pool_type="avg", name="global_pool")
    flatten = mx.sym.Flatten(data=pool, name="flatten")
    # # add quantize for fc
    # fc = mx.symbol.FullyConnected(data=flatten, num_hidden=num_classes, name='fc')
    fc = quant_fc(name='fc', data=flatten, num_hidden=num_classes, 
                  quant_mode=quant_mode, grad_mode=grad_mode, delay_quant=delay_quant, is_weight_perchannel=is_weight_perchannel, dict_shapes=dict_shapes)

    softmax = mx.symbol.SoftmaxOutput(data=fc, name='softmax')
    return softmax
