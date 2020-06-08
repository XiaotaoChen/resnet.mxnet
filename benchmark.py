import mxnet as mx
import argparse
from data.imagenet import SyntheticDataIter
import time
from symbol import *
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Train Faster R-CNN network')
    # general
    parser.add_argument('--kv_store', help='the kv-store type', type=str)
    parser.add_argument("--gpu_num", help="gpu number", type=int)
    parser.add_argument("--batch_size", help="batch size", type=int)
    parser.add_argument("--benchmark", help="benchmark or real train", type=str)
    parser.add_argument("--data_path", help="rec image path", type=str)

    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()
    kv_type = args.kv_store
    kv = mx.kvstore.create(kv_type)
    ctx = [mx.gpu(i) for i in range(args.gpu_num)]
    network = "resnet"
    batch_size = args.batch_size
    benchmark = eval(args.benchmark)
    rec_path = args.data_path
    
    num_classes = 1000
    units_dict = {"18": [2, 2, 2, 2],
                  "34": [3, 4, 6, 3],
                  "50": [3, 4, 6, 3],
                  "101": [3, 4, 23, 3],
                  "152": [3, 8, 36, 3]}
    num_stage = 4
    filter_list = [64, 256, 512, 1024, 2048]
    bottle_neck = True
    data_type = 'float32'

    sym = eval(network)(units=units_dict["50"],
                                      num_stage=num_stage,
                                      filter_list=filter_list,
                                      num_classes=num_classes,
                                      data_type=data_type,
                                      bottle_neck=bottle_neck)
    
    data_names = ('data',)
    label_names = ('softmax_label',)
    image_shape = [3, 224, 224]
    data_shapes = [('data', tuple([batch_size] + image_shape))]
    label_shapes = [('softmax_label', (batch_size,))]

    mx_mod = mx.mod.Module(symbol=sym, context=ctx, data_names=data_names, label_names=label_names)
    mx_mod.bind(for_training=True, data_shapes=data_shapes, label_shapes=label_shapes)
    mx_mod.init_params()
    mx_mod.init_optimizer()
    
    if benchmark:
        train_loader = SyntheticDataIter(num_classes, tuple([batch_size] + image_shape), 100, np.float32)
    else:
        train_loader = mx.io.ImageRecordIter(
            path_imgrec         = rec_path,
            label_width         = 1,
            data_name           = 'data',
            label_name          = 'softmax_label',
            data_shape          = image_shape,
            batch_size          = batch_size,
            pad                 = 0,
            fill_value          = 127,
            random_resized_crop = True,
            max_random_area     = 1.0,
            min_random_area     = 0.08,
            max_aspect_ratio    = 4.0 / 3.0,
            min_aspect_ratio    = 3.0 / 4.0,
            brightness          = 0.4,
            contrast            = 0.4,
            saturation          = 0.4,
            mean_r              = 123.68,
            mean_g              = 116.28,
            mean_b              = 103.53,
            std_r               = 58.395,
            std_g               = 57.12,
            std_b               = 57.375,
            pca_noise           = 0.1,
            scale               = 1,
            inter_method        = 2,
            rand_mirror         = True,
            shuffle             = True,
            shuffle_chunk_size  = 4096,
            preprocess_threads  = 16,
            prefetch_buffer     = 16,
            num_parts           = 1,
            part_index          = 0)
        train_loader = mx.io.ResizeIter(train_loader, 100)

    tic = time.time()
    for idx, data in enumerate(train_loader):
        mx_mod.forward(data, is_train=True)
        mx_mod.backward()
        mx_mod.update()
        mx.nd.waitall()
        if idx >= 5 and idx%10 == 0:
            interval = time.time() - tic
            print("{} {} kv type:{} benchmark:{} {} samples/s".format(idx, network, kv_type, benchmark, batch_size / interval))
        tic = time.time()
    print("*********** benchmark is done *************")



