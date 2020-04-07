import os
from mms import mms
from datetime import datetime
import shutil

from .graph_optimize import mergebn_for_deploy, attach_quantize_node
from .mms_services import upload_model, get_models_benchmark_id
from .split_trained_scale import split_int8_params
from .match_trt_scale_with_trained_scale import match_trt_with_quant_training

def upload_model_to_mms(name, model_type, extra_model_tags, file_list, pAutoPipeline):
    # to upload model to mms
    model_tags = [ "m_" + tag for tag in pAutoPipeline.model_tags]
    # model_tags.append("m_fp32")
    model_tags = model_tags + extra_model_tags.copy()
    data_tags = pAutoPipeline.TrainData.tags.copy()
    model_tags = data_tags.copy() + model_tags

    time_str = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    print("name:{}\nfile_list:{}\ndata tags:{}\nmodel_tags:{}\ntime_str:{}".format(
                    name, file_list, pAutoPipeline.TrainData.tags, model_tags, time_str))
    result = upload_model(name, file_list, model_type, model_tags, time_str)
    return result

def fetch_model_benchmark_from_mms(model_type, extra_model_tags, extra_benchmark_tags, pAutoPipeline):
    model_tags = [ "m_" + tag for tag in pAutoPipeline.model_tags]
    model_tags = pAutoPipeline.TestData.tags.copy() + extra_model_tags + model_tags
    benchmark_tags = model_tags.copy() + extra_benchmark_tags.copy()
    models, benchmark_id = get_models_benchmark_id(model_type, model_tags, benchmark_tags)
    print("************* auto test *************")
    print("model type:{}".format(model_type))
    print("model tags:{}".format(model_tags))
    print("benchmark_tags:{}".format(benchmark_tags))
    print("query benchmark id:{}, len(models):{}".format(benchmark_id, len(models)))
    return models, benchmark_id

def write_info_to_save_path(write_info, write_path='/root/.auto_train/lane.txt'):
    # write save path to special dir for int8 train
    if os.path.exists(write_path):
        os.remove(write_path)
    base_dir = os.path.dirname(write_path)
    os.makedirs(base_dir, exist_ok=True)
    with open(write_path, 'w') as f:
        f.write(write_info)

def delete_save_path(write_path='/root/.auto_train/lane.txt'):
    os.remove(write_path)

def write_info_by_dicts(dicts, file_path):
    write_info = "";
    for k, v in dicts.items():
        write_info += k + ":" + v + "\n"
    write_info_to_save_path(write_info, file_path)
    return write_info

def read_info_by_keys(keys, file_path):
    result = dict()
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            key, val = line.split(":")
            val = val.replace("\n", "")
            if key in keys:
                result[key] = val;
    return result;

def generate_mergebn_quantized_model(sym, arg_params, aux_params, pQuantize, sym_mode='test'):
    # merge bn
    if pQuantize.mergebn:
        sym, arg_params, aux_params = mergebn_for_deploy(sym, arg_params, aux_params)
    # attach quantize node
    assert sym_mode in ["train", "test"], "{} must be train or test".format(sym_mode)
    if sym_mode == "train":
        worker_data_shape = dict(pQuantize.data_shape + pQuantize.label_shape)
    else:
        worker_data_shape = dict(pQuantize.data_shape)
    _, out_shape, _ = sym.get_internals().infer_shape(**worker_data_shape)
    out_shape_dictoinary = dict(zip(sym.get_internals().list_outputs(), out_shape))
    sym = attach_quantize_node(sym, out_shape_dictoinary, pQuantize.WeightQuantizeParam, 
                                pQuantize.ActQuantizeParam, pQuantize.quantized_op, pQuantize.skip_quantize_counts,
                                pQuantize.visit_quantize_counts, pQuantize.special_node_infos)
    return sym, arg_params, aux_params

def get_param_sym(model_dir, platform, data_type):
    assert platform in ["mxnet", "trt"]
    assert data_type in ["fp32", "int8"]

    files = os.listdir(model_dir)
    param_path = None
    sym_path = None
    trained_scale_path = None
    if platform == "mxnet" or data_type == "fp32":
        for file in files:
            if file.endswith('.params'):
                param_path = os.path.join(model_dir, file)
            if file.endswith('-test.json'):
                sym_path = os.path.join(model_dir, file)
    else:
        for file in files:
            if file.endswith('.params'):
                param_path = os.path.join(model_dir, file)
            if file.endswith('-release.json'):
                sym_path = os.path.join(model_dir, file)
        # split int8 trained scale for trt int8
        model_prefix, epoch, trained_scale_path = split_int8_params(param_path, save_prefix='mx_int8')
        param_path = model_prefix + "-{:04d}.params".format(epoch)
    return param_path, sym_path, trained_scale_path


def replace_trained_scale(calib_json_path, trained_scale_path):
    replaced_json_path = match_trt_with_quant_training(calib_json_path, trained_scale_path)
    # backup source json
    shutil.move(calib_json_path, calib_json_path + '.bak')
    shutil.move(replaced_json_path, calib_json_path)

