from easydict import EasyDict as edict

config = edict()

config.gpu_list = [0, 1, 2, 3, 4, 5, 6, 7]
config.dataset = "imagenet"
config.model_prefix = "resnet50"
config.network = "resnet" # resnet  test_symbol
config.depth = 50

# data
config.data_dir = '/data/ILSVRC2012'
config.batch_per_gpu = 64
config.batch_size = config.batch_per_gpu * len(config.gpu_list)
config.kv_store = 'local'

# optimizer
config.lr = 3.2
config.wd = 0.0001
config.momentum = 0.9
config.lr_step = [40, 60, 80]
config.lr_factor = 0.1
config.begin_epoch = 0
config.num_epoch = 90
config.frequent = 20
# for distributed training
config.warmup_lr = 0.0
config.warm_epoch = 0
config.lr_scheduler = 'MultiFactor'
config.optimizer = 'sgd'
# set image_shape for io and network
config.image_shape = [3, 224, 224]
config.benchmark = 1
config.num_group = 64
config.data_type = 'float32'
config.grad_scale = 1.0
# using horovod or not
config.use_horovod = 0





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