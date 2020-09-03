# mxnet version: https://github.com/huangzehao/incubator-mxnet-bk
mxnet_path = '../incubator-mxnet/python/'
# gpu_list = [0, 1, 2, 3]
gpu_list = [4, 5, 6, 7]

dataset = "imagenet"
model_prefix = "resnet18"
network = "resnet"
depth = 18
model_load_prefix = model_prefix
model_load_epoch = 0
retrain = False

# data
data_dir = '/mnt/truenas/scratch/xiaotao.chen/dataset/imagenet/imagenet_data_new'
batch_size = 64
batch_size *= len(gpu_list)
kv_store = 'device'

# optimizer
lr = 0.1
wd = 0.0001
momentum = 0.9
if dataset == "imagenet":
    lr_step = [30, 60, 90]
else:
    lr_step = [120, 160, 240]
lr_factor = 0.1
begin_epoch = model_load_epoch if retrain else 0
num_epoch = 100
frequent = 50

# network config
if dataset == "imagenet":
    num_classes = 1000
    units_dict = {"18": [2, 2, 2, 2],
                  "34": [3, 4, 6, 3],
                  "50": [3, 4, 6, 3],
                  "101": [3, 4, 23, 3],
                  "152": [3, 8, 36, 3]}
    units = units_dict[str(depth)]
    if depth >= 50:
        filter_list = [64, 256, 512, 1024, 2048]
        bottle_neck = True
    else:
        filter_list = [64, 64, 128, 256, 512]
        bottle_neck = False
    num_stage = 4

# quantization setting
quant_begin_epoch = num_epoch
quant_end_epoch = quant_begin_epoch + 2
quant_lr = lr / 10
quantized_op = ("Convolution", "FullyConnected", "Deconvolution",)
skip_quantize_counts = {} # {"Convolution": 0, "FullyConnected":0}
quantize_counts = {} # {"Convolution": 1000, "FullyConnected":1000}

weight_setting = {
    "quantize_op_name": "Quantization_int8",
    "init_value": None,
    "attrs": {
        "nbits": 7,
        "delay_quant": 0,
        "ema_decay": 0.99,
        "grad_mode": "ste",
        "is_weight": True,
        "is_weight_perchannel": True,
        "fix_act_scale": False,
        "quant_mode": "minmax",
    },
}
act_setting = {
    "quantize_op_name": "Quantization_int8",
    "init_value": None,
    "attrs": {
        "nbits": 7,
        "delay_quant": 0,
        "ema_decay": 0.99,
        "grad_mode": "ste",
        "is_weight": False,
        "is_weight_perchannel": False,
        "fix_act_scale": False,
        "quant_mode": "minmax",
    },
}
