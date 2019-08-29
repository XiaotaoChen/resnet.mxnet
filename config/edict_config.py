from easydict import EasyDict as edict

config = edict()

# mxnet version: https://github.com/huangzehao/incubator-mxnet-bk
config.gpu_list = [0, 1, 2, 3, 4, 5, 6, 7]
#config.gpu_list = [0, 1, 2, 3]
config.platform = "aliyun"
config.dataset = "imagenet" # imagenet or cifar10
config.network = "mobilenet_int8"
config.depth = 50 if config.dataset == 'imagenet' else 50
config.model_load_epoch = 80
# config.model_prefix = config.network + '_' + config.dataset
config.model_prefix = config.network + '_' + config.dataset + "_retrain_" + str(config.model_load_epoch) + '_perchannel'
config.model_load_prefix = 'mobilenet/mobilenet'  # 'resnet50_new/resnet_imagenet'
config.retrain = True
config.use_global_stats=False
config.fix_gamma=False
# for int8 training
config.quant_mod = 'minmax'
config.delay_quant = 0
config.allow_missing = True
config.is_weight_perchannel = True

# data
if config.platform == 'truenas':
    config.data_dir = '/mnt/truenas/scratch/xiaotao.chen/dataset/imagenet/ILSVRC2012' if config.dataset == 'imagenet' \
        else '/mnt/truenas/scratch/xiaotao.chen/dataset/cifar10'
else:
    config.data_dir = '/mnt/tscpfs/bigfile/data/ILSVRC2012' if config.dataset == 'imagenet' \
        else '/mnt/tscpfs/xiaotao.chen/dataset/cifar10'
config.batch_per_gpu = 64
config.batch_size = config.batch_per_gpu * len(config.gpu_list)
config.kv_store = 'local'

# optimizer
config.lr = 0.1 * config.batch_per_gpu * len(config.gpu_list) / 256
config.wd = 0.0001
config.momentum = 0.9
config.multi_precision = True
if config.dataset == "imagenet":
    config.lr_step = [30, 60, 90]
    config.num_epoch = 100
else:
    config.lr_step = [120, 160, 240]
    config.num_epoch = 300
config.lr_factor = 0.1
config.begin_epoch = config.model_load_epoch if config.retrain else 0
config.frequent = 20
# for distributed training
if config.lr > 0.1 and config.retrain == False:
    config.warmup = True
else:
    config.warmup = False
config.warmup_lr = 0.1
config.warm_epoch = 5
config.lr_scheduler = 'warmup'
config.optimizer = 'sgd'
# set image_shape for io and network
config.benchmark = 0
config.num_group = 64
config.data_type = 'float32'
config.grad_scale = 1.0
config.data_nthreads = 16
config.use_multiple_iter = False
config.use_dali_iter = False
config.memonger = False

# network config
if config.dataset == "imagenet":
    config.num_classes = 1000
    config.image_shape = [3, 224, 224]
    config.num_stage = 4
    config.units_dict = {"18": [2, 2, 2, 2],
                  "34": [3, 4, 6, 3],
                  "50": [3, 4, 6, 3],
                  "101": [3, 4, 23, 3],
                  "152": [3, 8, 36, 3]}
    config.units = config.units_dict[str(config.depth)]
    if config.depth >= 50:
        config.filter_list = [64, 256, 512, 1024, 2048]
        config.bottle_neck = True
    else:
        config.filter_list = [64, 64, 128, 256, 512]
        config.bottle_neck = False
elif config.dataset == "cifar10":
    config.num_classes = 10
    config.image_shape = [3, 32, 32]
    config.num_stage = 3
    # depth should be one of 110, 164, 1001,...,which is should fit (args.depth-2)%9 == 0
    if ((config.depth - 2) % 9 == 0 and config.depth >= 164):
        per_unit = [int((config.depth - 2) / 9)]
        config.filter_list = [16, 64, 128, 256]
        config.bottle_neck = True
    elif ((config.depth - 2) % 6 == 0 and config.depth < 164):
        per_unit = [int((config.depth - 2) / 6)]
        config.filter_list = [16, 16, 32, 64]
        config.bottle_neck = False
    else:
        raise ValueError("no experiments done on detph {}, you can do it youself".format(args.depth))
    config.units = per_unit*3
else:
    raise ValueError("do not support {} yet".format(config.dataset))
