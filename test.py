import os
import sys
# import config
from config.edict_config import config

import mxnet as mx
from data import imagenet_iterator

from symbol import *

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Test Detection')
    parser.add_argument('--model_prefix', help='model prefix', type=str)
    parser.add_argument("--epoch", help="epoch", type=int)

    args = parser.parse_args()
    return args.model_prefix, args.epoch

def main(config):
    model_prefix, epoch = parse_args()
    symbol, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, epoch)

    model = mx.model.FeedForward(symbol, mx.gpu(0), arg_params=arg_params, aux_params=aux_params)
    kv = mx.kvstore.create(config.kv_store)
    _, val, _ = imagenet_iterator(data_dir=config.data_dir,
                                  batch_size=config.batch_size,
                                  kv=kv,
                                  image_shape=tuple(config.image_shape))
    print(model.score(val))


if __name__ == '__main__':
    main(config)
