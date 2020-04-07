import mxnet as mx
import argparse
import numpy as np
import pickle
import os

def get_params(params_path):
    save_dict = mx.nd.load(params_path)
    arg_params = {}
    aux_params = {}
    quantize_int8_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            if "minmax" in name:
                quantize_int8_params[name] = v
            else:
                aux_params[name] = v

    return arg_params, aux_params, quantize_int8_params

def save_split_params(arg_params, aux_params, quant_params, save_prefix):
    save_dict = {('arg:%s' % k): v.as_in_context(mx.cpu()) for k, v in arg_params.items()}
    save_dict.update({('aux:%s' % k): v.as_in_context(mx.cpu()) for k, v in aux_params.items()})
    mx.nd.save(save_prefix + "_arg_aux.params", save_dict)
    mx.nd.save(save_prefix + "_trained_scale.params", quant_params)

def split_int8_params(params_path, save_prefix):
    arg_params, aux_params, quant_params = get_params(params_path)
    epoch_str = os.path.basename(params_path)[-11:-7]
    abs_save_prefix = os.path.join(os.path.dirname(params_path), save_prefix)

    save_dict = {('arg:%s' % k): v.as_in_context(mx.cpu()) for k, v in arg_params.items()}
    save_dict.update({('aux:%s' % k): v.as_in_context(mx.cpu()) for k, v in aux_params.items()})
    
    model_prefix = abs_save_prefix + '_arg_aux'
    epoch = int(epoch_str)
    trained_scale_path = abs_save_prefix + "_trained_scale-{}.params".format(epoch_str)

    mx.nd.save(model_prefix + "-{}.params".format(epoch_str), save_dict)
    mx.nd.save(trained_scale_path, quant_params)

    return model_prefix, epoch, trained_scale_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='split trained int8 params from the trained model')
    parser.add_argument('--params_path', type=str, help='the params file path')
    parser.add_argument('--save_prefix', type=str, help='save path prefix')
    args = parser.parse_args()
    arg, aux, quants = get_params(args.params_path)
    save_split_params(arg, aux, quants, args.save_prefix)