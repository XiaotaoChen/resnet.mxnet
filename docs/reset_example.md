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


### upload model file and benchmark result to mms

image: artifactory.tusimple.ai/docker-rnd/cxt/pipeline-base-lidar0108/cuda10.0-cudnn7:pipeline0407

base on above  `train/test` flow,  we can upload `model file` to `mms`  when training is done. and fetch `model file` from `mms` to benchmark models queried from `mms`, and upload `benchmark result` to `mms`.

**pipeline script**

includes train/test scripts

```shell
./scripts/infra_pipeline.sh
```

### detail info

#### upload model to mms

in `infra_train.py`,  when training is done,  it will construct   model info, such as `file list`, `model_name`, `model_type` and `model tags`... than call `upload_model_to_mms` . 

in this example, model info as below:

```shell
name = "reset_for_infra"
model_type = "classification"
tags = ['imagenet', 'm_test', 'm_fp32', 'b_mx_fp32']
```

#### benchmark model and upload result to mms

in `test.py`,  we will fetch models from mms that meeting our querying conditions. in this scripts, we can config  benchmark `platform` and `data_type` , than it will construct a query condition to fetch model from mms.

in this example,  query info as below:

```shell
platform='mxnet'
data_type='fp32'

and the constructed querying info as below:

model type='classification'
model tags=['imagenet', 'm_fp32', 'm_test']
benchmark_tags=['imagenet', 'm_fp32', 'm_test', 'b_mx_fp32']
```

according above info, it will fetch model and benchmark. then upload result to `mms`

### outputs of upload model and benchmark

**upload model log info**

```shell
name:reset_for_infra
file_list:['/mnt/truenas/scratch/xiaotao.chen/outputs/infra/resnet/./model/resnet50-0001.params', '/mnt/truenas/scratch/xiaotao.chen/outputs/infra/resnet/./model/resnet50-test.json']
data tags:['imagenet']
model_tags:['imagenet', 'm_test', 'm_fp32']
time_str:2020-04-07-11:48:27
success upload model, model id:a3e853de-7882-11ea-8cef-32aad98aa46c
```

**fetch model and upload benchmark result**

```shell
model type:classification
model tags:['imagenet', 'm_fp32', 'm_test']
benchmark_tags:['imagenet', 'm_fp32', 'm_test', 'b_mx_fp32']
query benchmark id:e47062c4-7879-11ea-949b-32aad98aa46c, len(models):1
model id:a3e853de-7882-11ea-8cef-32aad98aa46c, download dir:/root/.mms/a3e853de-7882-11ea-8cef-32aad98aa46c, param_path:/root/.mms/a3e853de-7882-11ea-8cef-32aad98aa46c/resnet50-0001.params, sym_path:/root/.mms/a3e853de-7882-11ea-8cef-32aad98aa46c/resnet50-test.json, trained_scale_path:None
upload benchmark 0:
model_id:a3e853de-7882-11ea-8cef-32aad98aa46c, result_dict:{'acc': 0.0014186381074168797}
fallback:{'err_msg': '', 'success': True}
```

### you can query result from mms web
