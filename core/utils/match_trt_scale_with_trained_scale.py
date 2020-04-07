import mxnet as mx 
import json
import argparse

def match_trt_with_quant_training(trt_json_path, quant_params_path):
    quant_params = mx.nd.load(quant_params_path)
    act_minmaxs = {}
    for k,v in quant_params.items():
        if "weight" not in k:
            act_minmaxs[k] = v.asnumpy()[0]
    trt_json = json.load(open(trt_json_path, 'r'))
    match_count = 0
    match_keys = []
    lines={}
    for topic, calibs in trt_json.items():
        for k, v in calibs.items():
            lines[k] = v["scale"]
            trained_name = k + "_minmax"
            if trained_name in act_minmaxs.keys():
                match_count += 1
                match_keys.append(trained_name)
                print("{} : {} calib vs trained: {} vs {}".format(match_count, trained_name, v["scale"], 
                                                                  act_minmaxs[trained_name]))
                v["scale"] = float(act_minmaxs[trained_name])
                v["max"] = float(act_minmaxs[trained_name])
                calibs[k] = v
        trt_json[topic] = calibs
    
    print("len(trained params):{}, len(calib):{}".format(len(act_minmaxs), len(lines)))
    for k,v in act_minmaxs.items():
        print("{}:{}".format(k, v))
    for k,v in lines.items():
        trained_name = k + "_minmax"
        if trained_name not in act_minmaxs.keys():
            print("{}:{} not in trained scales".format(k, v))

    for act_minmax in act_minmaxs.keys():
        if act_minmax not in match_keys:
            print("trained scale: {} not in match to trt".format(act_minmax))

    save_path = trt_json_path.replace(".json", "_replace_trained.json")
    with open(save_path, 'w') as f:
        f.write(json.dumps(trt_json, indent=2))
    # for k,v in lines.items():
    #     print("{}:{}".format(k, v))
    return save_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='split trained int8 params from the trained model')
    parser.add_argument('--trt_calib_file', type=str, help='calibrate file path created by trt')
    parser.add_argument('--trained_scale_params', type=str, help='trained scale params path')
    args = parser.parse_args()

    trt_json_path = args.trt_calib_file
    trained_scale_path = args.trained_scale_params
    match_trt_with_quant_training(trt_json_path, trained_scale_path)