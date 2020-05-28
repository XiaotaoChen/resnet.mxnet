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

from .operator.QIL import *
from .operator.QIL_V2 import *
from .operator.QIL_V3 import *
from .operator.PACT import * # PACT_PY, DoReFa_PY
from .operator.WNQ import *
from .operator.GDRQ import *
from .operator.quantization_int8 import *


FLOAT32_DTYPE = 0

def get_constant(value):
    init_str = '[\"constant\", {\"value\": ' + str(value) + '}]'
    return init_str


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
                        auxs[mmean_name][:] = 0.0
                        auxs[mvar_name][:] = 1.0
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

def fix_bn(symbol):
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
            if "use_global_stats" not in attrs.keys() or attrs["use_global_stats"] == "False":
                attrs["use_global_stats"] = "True"
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
    return outputs

def create_quant_node(var, setting):
    if setting is None:
        return var
    quantize_op_name = setting.quantize_op_name
    attrs = setting.attrs
    assert quantize_op_name in ("Quantization_int8", "QIL", "DoReFa_PY", "DoReFa_CXX", "PACT", "PACT_CXX", 
                                "WNQ", "GDRQ", "GDRQ_CXX", "Quantization_int8_PY")

    if quantize_op_name == "Quantization_int8":
        init_value = setting.init_value or 0
        minmax_var = mx.sym.var(name = var.name + "_minmax", init=mx.init.Constant(init_value))
        quanted_node = mx.sym.contrib.Quantization_int8(name=var.name, data=var, minmax=minmax_var, **attrs)
    elif quantize_op_name == "Quantization_int8_PY":
        init_value = setting.init_value or 0
        minmax_var = mx.sym.var(name = var.name + "_minmax", init=mx.init.Constant(init_value))
        quanted_node = mx.sym.Custom(name=var.name, data=var, minmax=minmax_var, **attrs, op_type="Quantization_int8_PY")
    elif quantize_op_name == "QIL":
        init_value = setting.init_value or 1.0
        pruning_var = mx.sym.var(name = var.name + "_pruning_point", init=get_constant(0), lr_mult=0.01, wd_mult=0)
        clipping_var = mx.sym.var(name = var.name + "_clipping_point", init=get_constant(init_value), lr_mult=0.01, wd_mult=0)
        gamma_var = mx.sym.var(name = var.name + "_gamma", init=mx.init.Constant(1.0))
        quanted_node = mx.sym.Custom(name=var.name, data=var, pruning_point=pruning_var, clipping_point=clipping_var,
                                     gamma=gamma_var, **attrs, op_type='QIL_PY')
    elif quantize_op_name == "DoReFa_PY":
        quanted_node = mx.sym.Custom(name=var.name, data=var, **attrs, op_type="DoReFa_PY")
    elif quantize_op_name == "DoReFa_CXX":
        quanted_node = mx.sym.contrib.DoReFa(name=var.name, data=var, **attrs)
    elif quantize_op_name == "PACT":
        init_value = setting.init_value or 8.0
        gamma_var = mx.sym.var(name = var.name + "_pact_gamma", init=get_constant(init_value))
        quanted_node = mx.sym.Custom(name=var.name, data=var, gamma=gamma_var, **attrs, op_type="PACT_PY")
    elif quantize_op_name == "PACT_CXX":
        init_value = setting.init_value or 8.0
        gamma_var = mx.sym.var(name = var.name + "_pact_gamma", init=get_constant(init_value))
        quanted_node = mx.sym.contrib.PACT(name=var.name, data=var, gamma=gamma_var, **attrs)
    elif quantize_op_name == "GDRQ":
        init_value = setting.init_value or 1.0
        alpha_var = mx.sym.Variable(name=var.name + "_alpha", init=mx.init.Constant(init_value), dtype="float32")
        quanted_node = mx.sym.Custom(name=var.name, data=var, alpha=alpha_var, **attrs, op_type="GDRQ_PY")
    elif quantize_op_name == "GDRQ_CXX":
        init_value = setting.init_value or 1.0
        alpha_var = mx.sym.Variable(name=var.name + "_alpha", init=mx.init.Constant(init_value), dtype="float32")
        quanted_node = mx.sym.contrib.GDRQ(name=var.name, data=var, alpha=alpha_var, **attrs)
    return quanted_node

def attach_quantize_node(symbol, out_shape_dict, weight_setting, act_setting, 
                         quantized_op=("Convolution", "FullyConnected", "Deconvolution"), skip_quantize_counts=None):
    """
    Adapted from https://github.com/dmlc/tvm/blob/master/python/tvm/relay/frontend/mxnet.py
    Instead of translating nnvm graph into TVM relay graph, we adapt the script to translate
    it back to mxnet graph.
    """
    assert symbol is not None
    assert weight_setting is not None
    # assert act_setting is not None
    jgraph = json.loads(symbol.tojson())
    jnodes = jgraph["nodes"]
    node_map = {}
    node_op_map = {}
    quantized_node_map = {}
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
            if skip_quantize_counts is not None and op_name in skip_quantize_counts.keys() and \
                visited_op_counts[op_name] <= skip_quantize_counts[op_name]:
                print("skip idx:{} {} on {}".format(visited_op_counts[op_name], op_name, node_name))
                quanted_children = children
            elif op_name in ("Convolution", "FullyConnected", "Deconvolution"):
                if len(children) == 2:
                    datavar, weightvar = children
                    biasvar = None
                else:
                    datavar, weightvar, biasvar = children
                data_name, weight_name = datavar.name, weightvar.name
                if data_name in quantized_node_map.keys():
                    print("{} has attached quantized node".format(data_name))
                    data_quanted = quantized_node_map[data_name]
                else:
                    data_quanted = create_quant_node(datavar, act_setting)
                    quantized_node_map[data_name] = data_quanted
                if weight_name in quantized_node_map.keys():
                    print("{} has attached quantized node".format(weight_name))
                    weight_quanted = quantized_node_map[weight_name]
                else:
                    weight_quanted = create_quant_node(weightvar, weight_setting)
                    quantized_node_map[weight_name] = weight_quanted
                print("attach quantize node for {} inputs:{}, {}".format(op_name, data_name, weight_name))
                quanted_children = [data_quanted, weight_quanted, biasvar]
            elif op_name in ("Concat", "Pooling", "add_n", "elemwise_add"):
                quant_names = [var.name for var in children]
                print("attach quantize node for {} inputs:{}".format(op_name, quant_names))
                quanted_children = [None] * len(children)
                for i, var in enumerate(children):
                    if var.name in quantized_node_map.keys():
                        print("{} has attached quantized node".format(var.name))
                        quanted_children[i] = quantized_node_map[var.name]
                    else:
                        quanted_var = create_quant_node(var, act_setting)
                        quantized_node_map[var.name] = quanted_var
                        quanted_children[i] = quantized_node_map[var.name]
            
            operator = eval("mx.sym." + op_name)
            res = operator(*quanted_children, **attrs, name=node_name)
            node_map[nid] = res
            node_op_map[nid] = [op_name]
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
    return outputs

def do_kurt(var, shape, lamba, kT, weight_count):
    cnt = 1
    for dim in shape:
        cnt *= dim
    mean = mx.sym.mean(name=var.name+"_mean", data=var)
    meaned_var = mx.sym.broadcast_sub(name=var.name+"_broadcast_sub", lhs=var, rhs=mean)
    norm_var = mx.sym.norm(name=var.name+"_norm", data=meaned_var)

    # meaned_std = 1.0/np.sqrt(cnt) * norm_var
    # kurt = 1.0/cnt * mx.sym.sum((mx.sym.broadcast_div(meaned_var, meaned_std))**4)
    # kurt_loss = lambd/num_layers * (kurt - kT)
    # return mx.sym.make_loss(kurt_loss, grad_scale=1)

    kurt_val = cnt * mx.sym.sum(name=var.name+"_sum", data=(mx.sym.broadcast_div(meaned_var, norm_var)) **4)
    kurt_loss = (kurt_val - kT)**2

    return mx.sym.make_loss(name=var.name+"_loss", data=kurt_loss, grad_scale= lamba / weight_count)

def attach_kurt_loss(symbol, out_shape_dict, lamba=1.0, kT=1.8, weight_count=1):
    print("kurt loss setting lambda:{}, kT:{}, weight_count:{}".format(lamba, kT, weight_count))
    jgraph = json.loads(symbol.tojson())
    jnodes = jgraph["nodes"]
    node_map = {}
    node_op_map = {}

    outputs = []

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

        # attach kurt loss
        for var in children:
            if var.name.endswith("_weight"):
                assert var.name in out_shape_dict.keys(), "{} Variable is not in shape_dict".format(var.name)
                print("attach kurt loss for: {}".format(var.name))
                outputs.append(do_kurt(var, shape=out_shape_dict[var.name], lamba=lamba, kT=kT, weight_count=weight_count))
    for e in jgraph["heads"]:
        outputs.append(node_map[e[0]][e[1]])

    # outputs = [node_map[e[0]][e[1]] for e in jgraph["heads"]]
    outputs = outputs[0] if len(outputs) == 1 else mx.sym.Group(outputs)
    return outputs

if __name__ == "__main__":
    sym = mx.sym.load("source.json")
    sym1, _, _ = merge_bn(sym, None, None, True)
