from easydict import EasyDict as edict

config = edict()

# mxnet version: https://github.com/huangzehao/incubator-mxnet-bk
config.mxnet_path = '../mxnet/python/'
config.gpu_list = [0, 1, 2, 3, 4, 5, 6, 7]
config.dataset = "imagenet"
config.model_prefix = "resnet50"
config.network = "resnet"
config.depth = 50
config.output_dir="/mnt/truenas/scratch/xiaotao.chen/outputs/infra/resnet"

# data
config.data_dir = '/mnt/truenas/scratch/xiaotao.chen/dataset/imagenet/imagenet_data_new'
config.batch_per_gpu = 64
config.batch_size = config.batch_per_gpu * len(config.gpu_list)
config.kv_store = 'local'

# optimizer
config.lr = 0.2
config.wd = 0.0001
config.momentum = 0.9
config.multi_precision = True
if config.dataset == "imagenet":
    config.lr_step = [40, 60, 80]
else:
    config.lr_step = [120, 160, 240]
config.lr_factor = 0.1
config.begin_epoch = 0
config.num_epoch = 90
config.frequent = 20
# for distributed training
config.warmup_lr = 0.0
config.warm_epoch = 5
config.lr_scheduler = 'MultiFactor'
config.optimizer = 'sgd'
config.islars = 0
config.lars_eta = 1.0
config.isdebug = 0
# set image_shape for io and network
config.image_shape = [3, 224, 224]
config.benchmark = 0
config.num_group = 64
config.data_type = 'float32'
config.grad_scale = 1.0
config.data_nthreads = 1
config.use_multiple_iter = False
config.use_dali_iter = False
config.memonger = False
# using horovod or not
config.use_horovod = 1





# network config
if config.dataset == "imagenet":
    config.num_classes = 1000
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
    config.num_stage = 4
