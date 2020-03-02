import os
import sys
import time
import argparse

import mxnet as mx
from data import imagenet_iterator
from data import get_data_rec


def main():
    parser = argparse.ArgumentParser(description='Test resnet network')
    # general
    parser.add_argument('--model_prefix', help='pretrained model prefix', type=str)
    parser.add_argument('--model_load_epoch', help='pretrained model epoch', type=int)
    parser.add_argument('--data_dir', help='dataset path', type=str)
    args = parser.parse_args()
    symbol, arg_params, aux_params = mx.model.load_checkpoint('./model/' + args.model_prefix, args.model_load_epoch)
    print("model prefix:{}\nmodel_load_epoch:{}".format(args.model_prefix, args.model_load_epoch))
    model = mx.model.FeedForward(symbol, mx.gpu(), arg_params=arg_params, aux_params=aux_params)
    # _, val, _ = imagenet_iterator(data_dir=args.data_dir,
    #                               batch_size=256,
    #                               num_workers=1,
    #                               rank=0,
    #                               image_shape=tuple([3, 224, 224]))
    _, val, _ = get_data_rec(data_dir=args.data_dir,
                             batch_size=256,
                             data_nthreads=16,
                             num_workers=1,
                             rank=0)
    start_time = time.time()
    print("%d epoch score: %.4f" % (args.model_load_epoch,model.score(val)))
    end_time = time.time()
    print("cost time:{}".format(end_time - start_time))

if __name__ == '__main__':
    main()
