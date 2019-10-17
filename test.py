import os
import sys
# import config
from config.edict_config import config

import mxnet as mx
from data import imagenet_iterator

from symbol import *



def main(config):
    _, arg_params, aux_params = mx.model.load_checkpoint('./experiments/' + config.model_load_prefix, config.model_load_epoch)

    if config.network == 'resnet':
        symbol = eval(config.network)(units=config.units,
                                      num_stage=config.num_stage,
                                      filter_list=config.filter_list,
                                      num_classes=config.num_classes,
                                      data_type=config.data_type,
                                      bottle_neck=config.bottle_neck,
                                      grad_scale=config.grad_scale,
                                      memonger=config.memonger,
                                      dataset_type=config.dataset)
    elif config.network == 'resnet_mxnet':
        symbol = eval(config.network)(units=config.units,
                                      num_stage=config.num_stage,
                                      filter_list=config.filter_list,
                                      num_classes=config.num_classes,
                                      data_type=config.data_type,
                                      bottle_neck=config.bottle_neck,
                                      grad_scale=config.grad_scale)
    elif config.network == 'resnext' or config.network == 'resnext_cyt':
        symbol = eval(config.network)(units=config.units,
                                      num_stage=config.num_stage,
                                      filter_list=config.filter_list,
                                      num_classes=config.num_classes,
                                      data_type=config.data_type,
                                      num_group=config.num_group,
                                      bottle_neck=config.bottle_neck)
    elif config.network == 'vgg16' or config.network == 'mobilenet' or config.network == 'shufflenet':
        symbol = eval(config.network)(num_classes=config.num_classes)

    if config.fix_bn:
        from core.graph_optimize import fix_bn
        print("********************* fix bn ***********************")
        symbol = fix_bn(symbol)

    if config.quantize_flag:
        assert config.data_type == "float32", "current quantization op only support fp32 mode."
        from core.graph_optimize import attach_quantize_node
        data_shapes = [('data', tuple([config.batch_per_gpu] + config.image_shape))]
        label_shapes = [('softmax_label', (config.batch_per_gpu,))]
        worker_data_shape = dict(data_shapes + label_shapes)
        _, out_shape, _ = symbol.get_internals().infer_shape(**worker_data_shape)
        out_shape_dictoinary = dict(zip(symbol.get_internals().list_outputs(), out_shape))
        symbol = attach_quantize_node(symbol, out_shape_dictoinary, config.quantize_op_name, 
                                      config.base_quant_attrs, config.quantized_op, config.skip_quantize_counts)
        # symbol.save("test_tmp.json")
        # raise NotImplementedError

    print("data dir:{}, image_shape:{}".format(config.data_dir, config.image_shape))
    model = mx.model.FeedForward(symbol, mx.gpu(0), arg_params=arg_params, aux_params=aux_params)
    kv = mx.kvstore.create(config.kv_store)
    _, val, _ = imagenet_iterator(data_dir=config.data_dir,
                                  batch_size=config.batch_per_gpu,
                                  kv=kv,
                                  image_shape=tuple(config.image_shape))
    # val = mx.io.ResizeIter(val, 1)
    print(model.score(val))


if __name__ == '__main__':
    main(config)
