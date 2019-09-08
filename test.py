import os
import sys
# import config
from config.edict_config import config

import mxnet as mx
from data import imagenet_iterator

from symbol import *
from symbol.quantization_int8_V2 import *



def main(config):
    symbol, arg_params, aux_params = mx.model.load_checkpoint('./model/' + config.model_load_prefix, config.model_load_epoch)
    symbol = eval(config.network)(num_classes=config.num_classes,
                                      quant_mod=config.quant_mod,
                                      delay_quant=config.delay_quant,
                                      is_weight_perchannel=config.is_weight_perchannel,
                                      total_params_path=config.total_params_path,
                                      quantize_flag=config.quantize_flag)
    model = mx.model.FeedForward(symbol, mx.gpu(0), arg_params=arg_params, aux_params=aux_params)
    kv = mx.kvstore.create(config.kv_store)
    _, val, _ = imagenet_iterator(data_dir=config.data_dir,
                                  batch_size=config.batch_size,
                                #   batch_size=1,
                                  kv=kv,
                                  image_shape=tuple(config.image_shape))
    # val = mx.io.ResizeIter(val, 1)
    print(model.score(val))


if __name__ == '__main__':
    main(config)
