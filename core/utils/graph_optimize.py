# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import json
import logging
import mxnet as mx
import numpy as np 
import pickle as pkl

FLOAT32_DTYPE = 0
QUANT_TYPES = ("Quantization_int8", "DoReFa", "PACT", "GDRQ", "FQN")

def is_quant_node(node):
    attrs = node.get("attrs", {})
    op_name = node["op"]
    for quant_type in QUANT_TYPES:
        if quant_type in op_name:
            return True
    if op_name == "Custom" and attrs["op_type"] in QUANT_TYPES:
        return True
    return False


def convert_class_to_dict(obj):
    if obj is None:
        return {}
    pr = {}
    for name in dir(obj):
        value = getattr(obj, name)
        if not name.startswith('__') and not callable(value):
            pr[name] = str(value)
    return pr

def get_constant(value):
    init_str = '[\"constant\", {\"value\": ' + str(value) + '}]'
    return init_str


def create_quant_node(var, setting, init_dicts=None, special_node_infos=None):
    quantize_op_name = setting.quantize_op_name
    if quantize_op_name == "skip":
        return var

    attrs = convert_class_to_dict(setting.attrs)

    if special_node_infos is not None and var.name in special_node_infos.keys():
        for k,v in special_node_infos[var.name].items():
            if k in attrs.keys():
                logging.info("replace {} attrs.{}:{} to {}".format(var.name, k, attrs[k], v))
                attrs[k] = str(v)

    assert quantize_op_name in QUANT_TYPES

    if quantize_op_name == "Quantization_int8":
        if init_dicts is not None and var.name in init_dicts.keys():
            init_value = init_dicts[var.name]
        else:
            init_value = setting.init_value or 0
        minmax_var = mx.sym.var(name = var.name + "_minmax", init=mx.init.Constant(init_value))
        quanted_node = mx.sym.contrib.Quantization_int8(name=var.name, data=var, minmax=minmax_var, **attrs)
    elif quantize_op_name == "Quantization_int8_PY":
        if init_dicts is not None and var.name in init_dicts.keys():
            init_value = init_dicts[var.name]
        else:
            init_value = setting.init_value or 0
        minmax_var = mx.sym.var(name = var.name + "_minmax", init=mx.init.Constant(init_value))
        quanted_node = mx.sym.Custom(name=var.name, data=var, minmax=minmax_var, **attrs, op_type="Quantization_int8_PY")
    elif quantize_op_name == "DoReFa":
        quanted_node = mx.sym.contrib.DoReFa(name=var.name, data=var, **attrs)
    elif quantize_op_name == "PACT":
        if init_dicts is not None and var.name in init_dicts.keys():
            init_value = init_dicts[var.name]
        else:
            init_value = setting.init_value or 8.0
        gamma_var = mx.sym.var(name = var.name + "_gamma", init=get_constant(init_value))
        quanted_node = mx.sym.contrib.PACT(name=var.name, data=var, gamma=gamma_var, **attrs)
    elif quantize_op_name == "GDRQ":
        if init_dicts is not None and var.name in init_dicts.keys():
            init_value = init_dicts[var.name]
        else:
            init_value = setting.init_value or 1.0
        alpha_var = mx.sym.Variable(name=var.name + "_alpha", init=mx.init.Constant(init_value), dtype="float32")
        quanted_node = mx.sym.contrib.GDRQ(name=var.name, data=var, alpha=alpha_var, **attrs)
    elif quantize_op_name == "FQN_PY":
        if init_dicts is not None and var.name in init_dicts.keys():
            init_value = init_dicts[var.name]
        else:
            init_value = setting.init_value or 0
        min_var = mx.sym.var(name = var.name + "_min", init=mx.init.Constant(init_value))
        max_var = mx.sym.var(name = var.name + "_max", init=mx.init.Constant(init_value))
        quanted_node = mx.sym.Custom(name=var.name, data=var, min=min_var, max=max_var, **attrs, op_type="FQN_PY")
    elif quantize_op_name == "FQN":
        if init_dicts is not None and var.name in init_dicts.keys():
            init_value = init_dicts[var.name]
        else:
            init_value = setting.init_value or 0
        min_var = mx.sym.var(name = var.name + "_min", init=mx.init.Constant(init_value))
        max_var = mx.sym.var(name = var.name + "_max", init=mx.init.Constant(init_value))
        quanted_node = mx.sym.contrib.FQN(name=var.name, data=var, min=min_var, max=max_var, **attrs)
    elif quantize_op_name == "COLL_INFO_PY":
        quanted_node = mx.sym.Custom(name=var.name, data=var, **attrs, op_type="COLL_INFO_PY")

    return quanted_node


def merge_bn(symbol, args, auxs, symbol_only=False):
    """
    Adapted from https://github.com/dmlc/tvm/blob/master/python/tvm/relay/frontend/mxnet.py
    Instead of translating nnvm graph into TVM relay graph, we adapt the script to translate
    it back to mxnet graph.
    """
    assert symbol is not None
    jgraph = json.loads(symbol.tojson())
    jnodes = jgraph["nodes"]
    node_map = {}
    node_op_map = {}

    for nid, node in enumerate(jnodes):
        # edges are [which_node, which_output, type(? not sure)]
        # mx.symbol has an attribute of __getitem__. sym[1] gives the second output
        children = [node_map[e[0]][e[1]] for e in node["inputs"]]
        attrs = node.get("attrs", {})
        node_name = node["name"]
        op_name = node["op"]
        if op_name == "null":
            attrs = dict({k:v for k, v in attrs.items() if k.startswith("__")})
            node_map[nid] = mx.sym.var(node_name, **attrs)
            node_op_map[nid] = ["Variable"]
        elif op_name == "BatchNorm":
            e = node["inputs"][0]
            _, gamma, beta, mmean, mvar = children
            gamma_name, beta_name, mmean_name, mvar_name = gamma.name, beta.name, mmean.name, mvar.name
            assert "gamma" in gamma_name
            assert "beta" in beta_name
            assert "moving_mean" in mmean_name
            assert "moving_var" in mvar_name
            eps = float(attrs["eps"])
            if attrs["use_global_stats"] == "True" and node_op_map[e[0]][e[1]] == "Convolution":
                if not symbol_only:
                    if (mmean_name) not in auxs:
                        logging.info("Can not find {}, merge the symbol only".format(node_name + "_moving_mean"))
                    else:
                        logging.info("Merging {}".format(node_name))
                        # modify beta before gamma since gamma is not depend on beta
                        args[beta_name] -= args[gamma_name] * auxs[mmean_name] / mx.nd.sqrt(eps + auxs[mvar_name])
                        args[gamma_name] /= mx.nd.sqrt(eps + auxs[mvar_name])
                        # expand for broadcasting
                        if args[gamma_name].ndim == 1:
                            args[gamma_name] = args[gamma_name].expand_dims(axis=0).expand_dims(axis=-1).expand_dims(axis=-1)
                            args[beta_name] = args[beta_name].expand_dims(axis=0).expand_dims(axis=-1).expand_dims(axis=-1)
                            auxs[mmean_name] = auxs[mmean_name].expand_dims(axis=0).expand_dims(axis=-1).expand_dims(axis=-1)
                            auxs[mvar_name] = auxs[mvar_name].expand_dims(axis=0).expand_dims(axis=-1).expand_dims(axis=-1)
                        # set mmean and mvar to identity to avoid fusing more than once in weight sharing
                        auxs[mmean_name][:] = mx.nd.zeros_like(auxs[mmean_name])
                        auxs[mvar_name][:] = mx.nd.ones_like(auxs[mvar_name])
                        # copy shared gamma and beta for each BN
                        args[node_name + "_gamma"] = args[gamma_name]
                        args[node_name + "_beta"] = args[beta_name]
                # BroadcastScale is needed
                gamma = mx.sym.var(node_name + "_gamma", shape=args[node_name + "_gamma"].shape)
                beta = mx.sym.var(node_name + "_beta", shape=args[node_name + "_beta"].shape)
                res = mx.sym.broadcast_add(mx.sym.contrib.BroadcastScale(data=children[0], scaler=gamma), beta)
            else:
                res = mx.sym.BatchNorm(*children, **attrs, name=node_name)
            node_map[nid] = res
            node_op_map[nid] = ["BatchNorm"]
        else:
            if op_name.startswith("_contrib_"):
                op_name = op_name.replace("_contrib_", "")
                operator = eval("mx.sym.contrib." + op_name)
            elif op_name.startswith("_"):
                operator = eval("mx.sym._internal." + op_name)
            else:
                operator = eval("mx.sym." + op_name)
            res = operator(*children, **attrs, name=node_name)
            node_map[nid] = res
            node_op_map[nid] = [op_name]

    outputs = [node_map[e[0]][e[1]] for e in jgraph["heads"]]
    outputs = outputs[0] if len(outputs) == 1 else mx.sym.Group(outputs)
    return outputs, args, auxs

def attach_quantize_node(symbol, out_shape_dict, weight_setting, act_setting, 
                         quantized_op=None, 
                         skip_quantize_counts=None,
                         visit_quantize_counts=None,
                         special_node_infos=None):
    """
    Adapted from https://github.com/dmlc/tvm/blob/master/python/tvm/relay/frontend/mxnet.py
    Instead of translating nnvm graph into TVM relay graph, we adapt the script to translate
    it back to mxnet graph.
    """
    assert symbol is not None
    assert weight_setting is not None
    assert act_setting is not None
    init_dicts = None
    if act_setting.init_dict_path is not None:
        init_dicts = pkl.load(open(act_setting.init_dict_path, "rb"))
        for k, v in init_dicts.items():
            print("{}: {}".format(k, v))

    quantized_op = quantized_op or ("Convolution", "FullyConnected", "Deconvolution",
                                    "Concat", "concat", "Pooling", "add_n", "elemwise_add")

    # weight_quant_attrs = convert_class_to_dict(weight_quant_attrs)
    # act_quant_attrs = convert_class_to_dict(act_quant_attrs)    

    jgraph = json.loads(symbol.tojson())
    jnodes = jgraph["nodes"]
    node_map = {}
    node_op_map = {}
    quantized_node_map = {}
    skip_quantize_counts = convert_class_to_dict(skip_quantize_counts)
    visit_quantize_counts = convert_class_to_dict(visit_quantize_counts)
    logging.info("skip quantize_count:{}".format(skip_quantize_counts))
    logging.info("visit_quantize_counts:{}".format(visit_quantize_counts))
    logging.info("special_node_infos:{}".format(special_node_infos))
    logging.info("weight setting:{}".format(convert_class_to_dict(weight_setting.attrs)))
    logging.info("act setting:{}".format(convert_class_to_dict(act_setting.attrs)))
    # print("skip quantize_count:{}".format(skip_quantize_counts))
    # print("visit_quantize_counts:{}".format(visit_quantize_counts))
    # print("special_node_infos:{}".format(special_node_infos))
    # print("weight setting:{}".format(convert_class_to_dict(weight_setting.attrs)))
    # print("act setting:{}".format(convert_class_to_dict(act_setting.attrs)))

    visited_op_counts = {"Convolution": 0, "FullyConnected": 0, "Deconvolution": 0, 
                          "Concat": 0, "Pooling": 0, "add_n": 0, "elemwise_add": 0}

    for nid, node in enumerate(jnodes):
        # edges are [which_node, which_output, type(? not sure)]
        # mx.symbol has an attribute of __getitem__. sym[1] gives the second output
        children = [node_map[e[0]][e[1]] for e in node["inputs"]]
        attrs = node.get("attrs", {})
        node_name = node["name"]
        op_name = node["op"]
        if op_name == "null":
            attrs = dict({k:v for k, v in attrs.items() if k.startswith("__")})
            assert node_name in out_shape_dict.keys(), "{} Variable is not in shape_dict".format(node_name)
            if "__shape__" not in attrs.keys():
                attrs["__shape__"] = out_shape_dict[node_name]
                attrs["__dtype__"] = FLOAT32_DTYPE
            node_map[nid] = mx.sym.var(node_name, **attrs)
            node_op_map[nid] = ["Variable"]
        elif op_name in quantized_op:
            visited_op_counts[op_name] += 1
            # the idx of quantized_op to skip
            if op_name in skip_quantize_counts.keys() and \
                visited_op_counts[op_name] <= int(skip_quantize_counts[op_name]):
                logging.info("skip idx:{} {} on {}".format(visited_op_counts[op_name], op_name, node_name))
                quanted_children = children
            elif op_name in visit_quantize_counts.keys() and visited_op_counts[op_name] > int(visit_quantize_counts[op_name]):
                logging.info("the visited quantize count has reach the visit count: {}:{}, visited_count:{}".format(op_name, 
                              visit_quantize_counts[op_name], visited_op_counts[op_name]))
                quanted_children = children
            elif op_name in ["Convolution", "FullyConnected", "Deconvolution"]:
                if len(children) == 2:
                    datavar, weightvar = children
                    biasvar = None
                else:
                    datavar, weightvar, biasvar = children
                data_name, weight_name = datavar.name, weightvar.name
                if data_name in quantized_node_map.keys():
                    logging.info("{} has attached quantized node".format(data_name))
                    data_quanted = quantized_node_map[data_name]
                else:
                    data_quanted = create_quant_node(datavar, act_setting, init_dicts, special_node_infos=special_node_infos)
                    quantized_node_map[data_name] = data_quanted

                if weight_name in quantized_node_map.keys():
                    logging.info("{} has attached quantized node".format(weight_name))
                    weight_quanted = quantized_node_map[weight_name]
                else:
                    weight_quanted = create_quant_node(weightvar, weight_setting, special_node_infos=special_node_infos)
                    quantized_node_map[weight_name] = weight_quanted
                logging.info("attach quantize node for {} inputs:{}, {}".format(op_name, data_name, weight_name))
                quanted_children = [data_quanted, weight_quanted, biasvar]
            elif op_name in ["Concat", "concat", "Pooling", "add_n", "elemwise_add"]:
                quant_names = [var.name for var in children]
                logging.info("attach quantize node for {} inputs:{}".format(op_name, quant_names))
                quanted_children = [None] * len(children)
                for i, var in enumerate(children):
                    if var.name in quantized_node_map.keys():
                        logging.info("{} has attached quantized node".format(var.name))
                        quanted_children[i] = quantized_node_map[var.name]
                    else:
                        quanted_var = create_quant_node(var, act_setting, init_dicts, special_node_infos=special_node_infos)
                        quantized_node_map[var.name] = quanted_var
                        quanted_children[i] = quantized_node_map[var.name]

            else:
                logging.info("Warning {} don't support quantization training currently.".format(op_name))
                quanted_children = children
            operator = eval("mx.sym." + op_name)
            res = operator(*quanted_children, **attrs, name=node_name)
            node_map[nid] = res
            node_op_map[nid] = [op_name]
        else:
            # print("Warning {} don't support quantization training currently.".format(op_name))
            if op_name.startswith("_contrib_"):
                op_name = op_name.replace("_contrib_", "")
                operator = eval("mx.sym.contrib." + op_name)
            elif op_name.startswith("_"):
                operator = eval("mx.sym._internal." + op_name)
            else:
                operator = eval("mx.sym." + op_name)
            res = operator(*children, **attrs, name=node_name)
            node_map[nid] = res
            node_op_map[nid] = [op_name]

    outputs = [node_map[e[0]][e[1]] for e in jgraph["heads"]]
    outputs = outputs[0] if len(outputs) == 1 else mx.sym.Group(outputs)
    return outputs

def attach_quantize_node_with_node_name(symbol, out_shape_dict, weight_setting, act_setting, 
                         op_name_dict=None):
    """
    Adapted from https://github.com/dmlc/tvm/blob/master/python/tvm/relay/frontend/mxnet.py
    Instead of translating nnvm graph into TVM relay graph, we adapt the script to translate
    it back to mxnet graph.
    """
    assert symbol is not None
    assert weight_setting is not None
    assert act_setting is not None
    init_dicts = None
    if act_setting.init_dict_path is not None:
        init_dicts = pkl.load(open(act_setting.init_dict_path, "rb"))
        for k, v in init_dicts.items():
            print("{}: {}".format(k, v))

    # weight_quant_attrs = convert_class_to_dict(weight_quant_attrs)
    # act_quant_attrs = convert_class_to_dict(act_quant_attrs)    

    jgraph = json.loads(symbol.tojson())
    jnodes = jgraph["nodes"]
    node_map = {}
    node_op_map = {}
    quantized_node_map = {}
    
    logging.info("weight setting:{}".format(convert_class_to_dict(weight_setting.attrs)))
    logging.info("act setting:{}".format(convert_class_to_dict(act_setting.attrs)))
    print("weight setting:{}".format(convert_class_to_dict(weight_setting.attrs)))
    print("act setting:{}".format(convert_class_to_dict(act_setting.attrs)))

    
    for nid, node in enumerate(jnodes):
        # edges are [which_node, which_output, type(? not sure)]
        # mx.symbol has an attribute of __getitem__. sym[1] gives the second output
        children = [node_map[e[0]][e[1]] for e in node["inputs"]]
        attrs = node.get("attrs", {})
        node_name = node["name"]
        op_name = node["op"]
        if op_name == "null":
            attrs = dict({k:v for k, v in attrs.items() if k.startswith("__")})
            # assert node_name in out_shape_dict.keys(), "{} Variable is not in shape_dict".format(node_name)
            if "__shape__" not in attrs.keys() and node_name in out_shape_dict.keys():
                attrs["__shape__"] = out_shape_dict[node_name]
                attrs["__dtype__"] = FLOAT32_DTYPE
            node_map[nid] = mx.sym.var(node_name, **attrs)
            node_op_map[nid] = ["Variable"]
            # if node_name in op_name_dict.keys() and node_name not in quantized_node_map.keys():
                # if node_name not in quantized_node_map.keys():
            if node_name in op_name_dict.keys():
                if node_name not in quantized_node_map.keys():
                    logging.info("variable attach special quantize node: {}, init value:{}".format(node_name, op_name_dict[node_name]["scale"]))
                    act_setting.init_value = op_name_dict[node_name]["scale"]
                    quantized_node_map[node_name] = create_quant_node(node_map[nid], act_setting)
                    node_map[nid] = quantized_node_map[node_name]
                else:
                    logging.info("attach special quantize node: {} has quantized".format(node_name))
                    node_map[nid] = quantized_node_map[node_name]
        else:
            # print("Warning {} don't support quantization training currently.".format(op_name))
            
            if op_name.startswith("_contrib_"):
                op_name = op_name.replace("_contrib_", "")
                operator = eval("mx.sym.contrib." + op_name)
            elif op_name.startswith("_"):
                operator = eval("mx.sym._internal." + op_name)
            else:
                operator = eval("mx.sym." + op_name)
            res = operator(*children, **attrs, name=node_name)
            node_map[nid] = res
            node_op_map[nid] = [op_name]
            if node_name in op_name_dict.keys():
                if node_name not in quantized_node_map.keys():
                    logging.info("attach special quantize node: {}, init value:{}".format(node_name, op_name_dict[node_name]["scale"]))
                    act_setting.init_value = op_name_dict[node_name]["scale"]
                    quantized_node_map[node_name] = create_quant_node(node_map[nid], act_setting)
                    node_map[nid] = quantized_node_map[node_name]
                else:
                    logging.info("attach special quantize node: {} has quantized".format(node_name))
                    node_map[nid] = quantized_node_map[node_name]


    outputs = [node_map[e[0]][e[1]] for e in jgraph["heads"]]
    outputs = outputs[0] if len(outputs) == 1 else mx.sym.Group(outputs)
    return outputs


def extract_activation_node(symbol, out_shape_dict, quantized_op=None, skip_quantize_counts=None):
    """
    Adapted from https://github.com/dmlc/tvm/blob/master/python/tvm/relay/frontend/mxnet.py
    Instead of translating nnvm graph into TVM relay graph, we adapt the script to translate
    it back to mxnet graph.
    """
    assert symbol is not None

    quantized_op = quantized_op or ("Convolution", "FullyConnected", "Deconvolution",
                                    "Concat", "concat", "Pooling", "add_n", "elemwise_add")

    # weight_quant_attrs = convert_class_to_dict(weight_quant_attrs)
    # act_quant_attrs = convert_class_to_dict(act_quant_attrs)    

    jgraph = json.loads(symbol.tojson())
    jnodes = jgraph["nodes"]
    node_map = {}
    node_op_map = {}
    skip_quantize_counts = convert_class_to_dict(skip_quantize_counts)
    logging.info("skip quantize_count:{}".format(skip_quantize_counts))

    visited_op_counts = {"Convolution": 0, "FullyConnected": 0, "Deconvolution": 0, 
                          "Concat": 0, "Pooling": 0, "add_n": 0, "elemwise_add": 0}

    quantized_fm_names = []

    for nid, node in enumerate(jnodes):
        # edges are [which_node, which_output, type(? not sure)]
        # mx.symbol has an attribute of __getitem__. sym[1] gives the second output
        children = [node_map[e[0]][e[1]] for e in node["inputs"]]
        attrs = node.get("attrs", {})
        node_name = node["name"]
        op_name = node["op"]
        if op_name == "null":
            attrs = dict({k:v for k, v in attrs.items() if k.startswith("__")})
            assert node_name in out_shape_dict.keys(), "{} Variable is not in shape_dict".format(node_name)
            if "__shape__" not in attrs.keys():
                attrs["__shape__"] = out_shape_dict[node_name]
                attrs["__dtype__"] = FLOAT32_DTYPE
            node_map[nid] = mx.sym.var(node_name, **attrs)
            node_op_map[nid] = ["Variable"]
        elif op_name in quantized_op:
            visited_op_counts[op_name] += 1
            # the idx of quantized_op to skip
            if skip_quantize_counts is not None and op_name in skip_quantize_counts.keys() and \
                visited_op_counts[op_name] <= int(skip_quantize_counts[op_name]):
                logging.info("skip idx:{} {} on {}".format(visited_op_counts[op_name], op_name, node_name))
            elif op_name in ["Convolution", "FullyConnected", "Deconvolution"]:
                if len(children) == 2:
                    datavar, weightvar = children
                    biasvar = None
                else:
                    datavar, weightvar, biasvar = children
                data_name, weight_name = datavar.name, weightvar.name
                if data_name in quantized_fm_names:
                    logging.info("{} has attached quantized node".format(data_name))
                else:
                    quantized_fm_names.append(data_name)
                    logging.info("extract activation node: {}".format(data_name))
            elif op_name in ["Concat", "concat", "Pooling", "add_n", "elemwise_add"]:
                quant_names = [var.name for var in children]
                logging.info("attach quantize node for {} inputs:{}".format(op_name, quant_names))
                for i, var in enumerate(children):
                    if var.name in quantized_fm_names:
                        logging.info("{} has attached quantized node".format(var.name))
                    else:
                        quantized_fm_names.append(var.name)
                        logging.info("extract activation node: {}".format(var.name))


            operator = eval("mx.sym." + op_name)
            res = operator(*children, **attrs, name=node_name)
            node_map[nid] = res
            node_op_map[nid] = [op_name]
        else:
            # print("Warning {} don't support quantization training currently.".format(op_name))
            if op_name.startswith("_contrib_"):
                op_name = op_name.replace("_contrib_", "")
                operator = eval("mx.sym.contrib." + op_name)
            elif op_name.startswith("_"):
                operator = eval("mx.sym._internal." + op_name)
            else:
                operator = eval("mx.sym." + op_name)
            res = operator(*children, **attrs, name=node_name)
            node_map[nid] = res
            node_op_map[nid] = [op_name]

    outputs = [node_map[e[0]][e[1]] for e in jgraph["heads"]]
    outputs = outputs[0] if len(outputs) == 1 else mx.sym.Group(outputs)
    # return outputs
    return quantized_fm_names

def mergebn_for_deploy(symbol, args, auxs):
    """
    Adapted from https://github.com/dmlc/tvm/blob/master/python/tvm/relay/frontend/mxnet.py
    Instead of translating nnvm graph into TVM relay graph, we adapt the script to translate
    it back to mxnet graph.
    """

    assert symbol is not None
    jgraph = json.loads(symbol.tojson())
    jnodes = jgraph["nodes"]
    node_map = {}
    node_op_map = {}

    # added by cxt
    nid2node = {}

    for nid, node in enumerate(jnodes):
        # edges are [which_node, which_output, type(? not sure)]
        # mx.symbol has an attribute of __getitem__. sym[1] gives the second output
        children = [node_map[e[0]][e[1]] for e in node["inputs"]]
        attrs = node.get("attrs", {})
        node_name = node["name"]
        op_name = node["op"]

        nid2node[nid] = node

        if op_name == "null":
            attrs = dict({k:v for k, v in attrs.items() if k.startswith("__")})
            node_map[nid] = mx.sym.var(node_name, **attrs)
            node_op_map[nid] = ["Variable"]

        elif op_name == "BatchNorm" and node_op_map[node["inputs"][0][0]][node["inputs"][0][1]] == "Convolution":
            e = node["inputs"][0]

            conv_nid = e[0]
            conv_node = nid2node[conv_nid]
            conv_attrs = conv_node.get("attrs", {}).copy()
            if  "no_bias" in conv_attrs.keys() and conv_attrs["no_bias"] == "True":
                conv_attrs["no_bias"] = "False"

            if args is not None and auxs is not None:
                _, gamma, beta, mmean, mvar = children
                gamma_name, beta_name, mmean_name, mvar_name = gamma.name, beta.name, mmean.name, mvar.name
                assert "gamma" in gamma_name
                assert "beta" in beta_name
                assert "moving_mean" in mmean_name
                assert "moving_var" in mvar_name
                eps = float(attrs["eps"])
                assert mmean_name in auxs.keys() and mvar_name in auxs.keys(), "{}/{} can't found mean/var name in auxs".format(mmean_name, mvar_name)
                # get conv weight
                conv_w_name = conv_node['name'] + '_weight'
                assert conv_w_name in args.keys(), "{} not in args".format(conv_w_name)
                # modify beta before gamma since gamma is not depend on beta
                args[beta_name] -= args[gamma_name] * auxs[mmean_name] / mx.nd.sqrt(eps + auxs[mvar_name])
                args[gamma_name] /= mx.nd.sqrt(eps + auxs[mvar_name])

                assert args[conv_w_name].shape[0] == args[beta_name].shape[0], "weight shape \
                vs bn_beta shape:{} vs {}".format(args[conv_w_name].shape, args[beta_name].shape)
        
                # logging.info("Merging {} and {}".format(conv_node['name'], node_name))
                print("Merging {} and {}".format(conv_node['name'], node_name))
                # update conv bias
                conv_bias_name = conv_node['name'] + '_bias'
                tmp_attrs = conv_node.get("attrs", {}).copy()
                if "no_bias" not in tmp_attrs.keys() or tmp_attrs["no_bias"] == "False":
                    assert conv_bias_name in args.keys()
                    args[conv_bias_name] = args[conv_bias_name] * args[gamma_name] + args[beta_name]
                elif tmp_attrs["no_bias"] == "True":
                    args[conv_bias_name] = args[beta_name].copy()
                # expand for broadcasting for conv weight
                args[gamma_name] = args[gamma_name].expand_dims(axis=-1).expand_dims(axis=-1).expand_dims(axis=-1)
                # multiple gamma to weight
                args[conv_w_name][:] = args[conv_w_name] * args[gamma_name]
                # delete mean,var in auxs, gamma in args
                del auxs[mmean_name], auxs[mvar_name], args[gamma_name]
                del args[beta_name]

            # create new conv with bias
            conv_children = [node_map[e[0]][e[1]] for e in conv_node["inputs"]]
            res = mx.sym.Convolution(*conv_children, **conv_attrs, name=conv_node['name'])
            node_map[nid] = res
            node_op_map[nid] = ["Convolution"]

        else:
            if op_name.startswith("_contrib_"):
                op_name = op_name.replace("_contrib_", "")
                operator = eval("mx.sym.contrib." + op_name)
            elif op_name.startswith("_"):
                operator = eval("mx.sym._internal." + op_name)
            else:
                operator = eval("mx.sym." + op_name)
            res = operator(*children, **attrs, name=node_name)
            node_map[nid] = res
            node_op_map[nid] = [op_name]

    outputs = [node_map[e[0]][e[1]] for e in jgraph["heads"]]
    outputs = outputs[0] if len(outputs) == 1 else mx.sym.Group(outputs)
    return outputs, args, auxs

if __name__ == "__main__":
    # sym = mx.sym.load("source.json")
    # sym1, _, _ = merge_bn(sym, None, None, True)

    from .load_model import load_checkpoint
    model_dir = "auto_faster_var_600_80_2020-03-04-11/"
    model_prefix = os.path.join(model_dir, 'checkpoint')
    sym_path = os.path.join(model_dir, 'checkpoint-train.json')
    epoch=36

    sym = mx.sym.load(sym_path)
    arg_params, aux_params = load_checkpoint(model_prefix, epoch)
    sym, arg_params, aux_params = mergebn_for_deploy(sym, arg_params, aux_params)

    mx.model.save_checkpoint(model_prefix+'_mergebn', epoch, sym, arg_params, aux_params)
