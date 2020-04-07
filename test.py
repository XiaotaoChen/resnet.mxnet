import os
import sys
import time
import argparse
from config.edict_config import config
import mxnet as mx
from data import imagenet_iterator
from data import get_data_rec

from core.utils.pipeline_utils import fetch_model_benchmark_from_mms
from core.utils.pipeline_utils import get_param_sym
from core.utils.mms_services import fetch_model_dir
from core.utils.mms_services import upload_benchmark_result

def evaluate(symbol, param_dict):
    arg_params = {}
    aux_params = {}
    for k, v in param_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v
    
    data_dir = config.data_dir
    model = mx.model.FeedForward(symbol, mx.gpu(), arg_params=arg_params, aux_params=aux_params)
    _, val, _ = get_data_rec(data_dir=data_dir,
                             batch_size=64,
                             data_nthreads=16,
                             num_workers=1,
                             rank=0)
    start_time = time.time()
    result = model.score(val)
    print("score: %.4f" % (result))
    end_time = time.time()
    print("cost time:{}".format(end_time - start_time))
    return {"acc": result}


def main():
    parser = argparse.ArgumentParser(description='Test resnet network')
    # general
    parser.add_argument("--platform", help="platform", type=str)
    parser.add_argument("--data_type", help="test data type: fp32 or int8", type=str)
    args = parser.parse_args()

    platform = args.platform
    data_type = args.data_type
    pAutoPipeline = config.AutoPipelineParam

    name = "resnet_for_infra"
    model_type = "classification"
    map2str={"mxnet":"mx", "trt":"trt"}

    extra_model_tags = ["m_" + data_type,]
    extra_benchmark_tags = ["b_" + map2str[platform] + "_" + data_type,]

    models, benchmark_id = fetch_model_benchmark_from_mms(model_type, extra_model_tags, extra_benchmark_tags, pAutoPipeline)

    for i, model in enumerate(models):
        model_id = model['_id']
        download_dir = fetch_model_dir(model_id)
        param_path, sym_path, trained_scale_path = get_param_sym(download_dir, platform=platform, data_type=data_type)
        print("model id:{}, download dir:{}, param_path:{}, sym_path:{}, trained_scale_path:{}".format(
               model_id, download_dir, param_path, sym_path, trained_scale_path))
        
        params_dict = mx.nd.load(param_path)
        sym = mx.sym.load(sym_path)

        result = evaluate(sym, params_dict)
        if len(result) == 0:
            print("there are something wrong with eval result:{}".format(result))
            exit(1)
        # upload benchmark result
        fallback = upload_benchmark_result(model_id, benchmark_id, result)
        print("upload benchmark {}:\nmodel_id:{}, result_dict:{}\nfallback:{}".format(i, model_id, result,fallback))

    

if __name__ == '__main__':
    main()
