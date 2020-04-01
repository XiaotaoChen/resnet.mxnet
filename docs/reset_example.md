## example of distributed training model for infra

**image:** artifactory.tusimple.ai/docker-rnd/cxt/pipeline-base-lidar0108/cuda10.0-cudnn7:dist_example

**workspace:**  `/root/resnet.mxnet`, enter this directory, and run below scripts.

### distributed training

**scripts:** 

```shell
./scripts/infra_horovodrun.sh
```

**please config your training setting in  `config/edict_config.py`** . for example:

```shell
config.model_prefix = "resnet50"   # the model file prefix
config.network = "resnet"    # network to trained
config.depth = 50   # network layers, other setting: 18, 34
config.output_dir="/mnt/truenas/scratch/xiaotao.chen/outputs/infra/resnet"  # the output dir.
```

according to this config: all outputs will save to `/mnt/truenas/scratch/xiaotao.chen/outputs/infra/resnet`, 

model file in `/mnt/truenas/scratch/xiaotao.chen/outputs/infra/resnet/model`

log file in `/mnt/truenas/scratch/xiaotao.chen/outputs/infra/resnet/log`

### test

**scripts:**

```shell
./scripts/infra_test.sh
```

**config the script to test your trained model** just modify `model_prefix` and `epoch` in `infra_test.sh`, for example:

```shell
#!/bin/bash
model_prefix="/mnt/truenas/scratch/xiaotao.chen/outputs/infra/resnet/trained_model/resnet"
epoch=90
python3 test.py \
   --model_prefix ${model_prefix} \
   --model_load_epoch ${epoch}
```

this script will test the model in `/mnt/truenas/scratch/xiaotao.chen/outputs/infra/resnet/trained_model` with param file named `reset-0090.params`, symbol file named `resnet-symbol.json`