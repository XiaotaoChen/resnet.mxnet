from easydict import EasyDict as edict

config = edict()

# mxnet version: https://github.com/huangzehao/incubator-mxnet-bk
# config.gpu_list = [0, 1, 2, 3, 4, 5, 6, 7]
config.gpu_list = [0, 1, 2, 3]
# config.gpu_list = [4, 5, 6, 7]
config.dataset = "imagenet" # imagenet , cifar10 , cifar100 
config.network = "resnet" # "resnet_cifar10"  # "cifar10_sym"  # "resnet" # "preact_resnet"
config.depth = 18
config.model_prefix = config.network + str(config.depth) + '_' + config.dataset
config.model_load_epoch =90
config.model_load_prefix = "experiments/resnet18_imagenet_0508/resnet18_imagenet"
config.retrain = True
config.allow_missing = True



# data
config.data_dir = "imagenet_data_new"
config.batch_per_gpu = 64
config.batch_size = config.batch_per_gpu * len(config.gpu_list)
config.kv_store = 'device'

# optimizer

if config.dataset == "cifar10":
    config.lr = 0.1 * config.batch_per_gpu * len(config.gpu_list) / 128
    config.wd = 0.0002
else:
    config.lr = 0.1 * config.batch_per_gpu * len(config.gpu_list) / 256
    config.wd = 0.0001
config.momentum = 0.9
config.multi_precision = True
if config.dataset == "imagenet":
    config.lr_step = [30, 60, 80]
    config.num_epoch = 90
    # config.lr_step = [30, 60, 85, 95]   # PACT
    # config.lr_step = [30, 70, 90]
    # config.num_epoch = 110
elif config.dataset == "cifar10":
    config.lr_step = [60, 120]
    config.num_epoch = 200
elif config.dataset == "cifar100":
    config.lr_step = [30, 60, 90]
    config.num_epoch = 100

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
        raise ValueError("no experiments done on detph {}, you can do it youself".format(config.depth))
    config.units = per_unit*3
elif config.dataset == "cifar100":
    config.num_classes = 100
    config.image_shape = [3, 28, 28]
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
else:
    raise ValueError("do not support {} yet".format(config.dataset))



# for quantize int8 training
config.quantize_flag = True
config.quantize_setting = {
# for GDRQ
    "weight":{
        "quantize_op_name": "GDRQ_CXX",
        "init_value": 1.0,
        "attrs": {
            "nbits": "4",
            "fix_alpha": "False",
            "group_size": "-1",
            "is_weight": "True",
            "lamda": "0.001",
            "delay_quant": "0",
            "ktimes": "3"
        }
    },
    "act":{
        "quantize_op_name": "GDRQ_CXX",
        "init_value": 11.0,
        "attrs": {
            "nbits": "4",
            "fix_alpha": "False",
            "group_size": "-1",
            "is_weight": "False",
            "lamda": "0.001",
            "delay_quant": "0",
            "ktimes": "3"
        }
    }
}

# config.quantized_op = ["Convolution", "FullyConnected", "Deconvolution","Concat", "Pooling", "add_n", "elemwise_add"]
config.quantized_op = ["Convolution", "FullyConnected", "Deconvolution"]
config.skip_quantize_counts = {"Convolution": 1, "FullyConnected": 1}
config.fix_bn = False
if config.quantize_flag is False:
    config.output_dir = "{}_0508".format(config.model_prefix)
else:
    config.num_epoch = 120
    config.output_dir = "{}_{}_w{}_act{}".format(config.model_prefix,
                                                config.quantize_setting["act"]["quantize_op_name"],
                                                config.quantize_setting["weight"]["attrs"]["nbits"], 
                                                config.quantize_setting["act"]["attrs"]["nbits"])
