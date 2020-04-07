import os
import shutil
from datetime import datetime
from mms import mms
from itertools import product
import mxnet as mx

def upload_model(name, file_list, model_type, tags, time_str):
    service = mms()
    kwargs = {'name': name,
              'created_by': 'Xiaotao Chen',
              'type': model_type,
              'model_file': file_list,
              'tags': tags,
              'datetime': time_str,
              'description': 'test for mms',
              'dataset_version': 'trained by roidbs in used_roidb.txt'}
    return service.upload_model(kwargs)

def append_model_file(model_id, to_append_files):
    service = mms()
    model = service.list_model_by_id(model_id)['result']
    model_name = model['name']
    result = service.update_model(model_id, 'Xiaotao Chen', {'model_file': to_append_files, 'name': model_name})
    return result

def create_benchmark(name, type, tags):
    service = mms()
    kwargs = {'name': name,
              'created_by': 'Xiaotao Chen',
              'type': type,
              'tags': tags}
    result_dict = service.create_benchmark(kwargs)
    if result_dict['success']:
        return result_dict['benchmark_id']
    else:
        return None

def fetch_model_dir(model_id):
    service = mms()
    result = service.fetch_model(model_id)
    if result['success'] is True:
        return result['result']
    else:
        print("can't fetch model:{}, err_msg:{}".format(model_id, result['err_msg']))
    return None

def rm_download_dir(download_dir):
    shutil.rmtree(download_dir)

def get_model_param_sym(model_id):
    download_dir = fetch_model_dir(model_id)
    if download_dir is None:
        return None, None
    files = os.listdir(download_dir)
    param_path = None
    sym_path = None
    for file in files:
        if file.endswith('.params'):
            param_path = file
        if file.endswith('-test.json'):
            sym_path = file
    # assert param_path is not None and sym_path is not None
    if param_path is None or sym_path is None:
        print("model:{} param_path or sym_path is None, ignore it".format(model_id))
        shutil.rmtree(download_dir)
        return None, None
    sym = mx.sym.load(os.path.join(download_dir,sym_path))
    param_dict = mx.nd.load(os.path.join(download_dir,param_path))

    rm_download_dir(download_dir)

    return param_dict, sym

def get_models_benchmark_id(type, model_tags, benchmark_tags):
    service = mms()
    # use kwargs in list benchmark the kwargs will become {'type': type}, \
    # make the wrong result in list_models
    # kwargs = {'type': type, 'tags': tags}

    # benchmarks = service.list_benchmark({'type': type, 'tags': benchmark_tags})['result']
    result = service.list_benchmark({'type': type, 'tags': benchmark_tags})
    if result['success'] == False:
        raise ValueError("list benchmark with type:{}, tags:{} failed, err msg:{}".format(type, benchmark_tags, result['err_msg']))
    benchmarks = result['result']
    assert len(benchmarks)==1, "benhcmark len:{} with type:{}, tags:{}".format(len(benchmarks), type, benchmark_tags)
    benchmark_id = benchmarks[0]['_id']

    # models = service.list_models({'type': type, 'tags': model_tags})['result']
    result = service.list_models({'type': type, 'tags': model_tags})
    if result['success'] == False:
        raise ValueError("list model with type:{}, tags:{} failed, err msg:{}".format(type, model_tags, result['err_msg']))
    models = result['result']

    # benchmarked_models = service.list_benchmark_by_id(benchmark_id)['result']
    result = service.list_benchmark_by_id(benchmark_id)
    if result['success'] == False:
        raise ValueError("list benchmark by id:{} failed, err msg:{}".format(benchmark_id, result['err_msg']))
    benchmarked_models = result['result']

    if 'result' not in benchmarked_models.keys():
        return models, benchmark_id
    filtered_models = []
    for model in models:
        if model['_id'] not in benchmarked_models['result'].keys():
            filtered_models.append(model)
    return filtered_models, benchmark_id

def upload_benchmark_result(model_id, benchmark_id, result_dict):
    service = mms()
    fallback = service.add_benchmark(model_id, benchmark_id, result_dict)
    return fallback


if __name__ == '__main__':
    name = '0214_test_m_int8_b_trt_int8'
    type = 'lidar-detection'
    tags = ['lidar-detection', 'roidb_list', 'm_test', 'm_int8', 'b_trt_int8']
    benchmark_info = create_benchmark(name, type, tags)
    print(benchmark_info)

    # file_list = ['/mnt/truenas/upload/xiaotao.chen/outputs/lidar_pipeline/auto_train/auto_faster_var_1024_80_old_data_2020-02-10-10/' + 
    #              'w_Quantization_int87_act_Quantization_int87_36e_46e/checkpoint-0038.params',
    #              '/mnt/truenas/upload/xiaotao.chen/outputs/lidar_pipeline/auto_train/auto_faster_var_1024_80_old_data_2020-02-10-10/' + 
    #              'w_Quantization_int87_act_Quantization_int87_36e_46e/checkpoint-release.json',
    #              '/mnt/truenas/upload/xiaotao.chen/outputs/lidar_pipeline/auto_train/auto_faster_var_1024_80_old_data_2020-02-10-10/' + 
    #              'w_Quantization_int87_act_Quantization_int87_36e_46e/checkpoint-test.json',
    # ]
    # time_str = datetime.now().strftime("%Y-%m-%d-%H")
    # model_info = upload_model(name, file_list, tags, time_str)
    # print(model_info)

