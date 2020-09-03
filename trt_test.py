import json
import os
import numpy as np
import tensorrt as trt

import mxnet as mx

import trtplus
from trtplus import tensor as T
from trtplus.calibrator import GeneratorEntropyCalibrator
from trtplus.converters.mxnet2trt import create_weight_dict, mxnet2trt

from trtplus.converters.mxnet2trt import MX_TYPE_TO_BUILDER
def dense_nchw(inputs, name, param, network, version):
    x = inputs[0]
    ndim = len(x.shape) + network.has_implicit_batch_dimension
    if ndim == 2:
        x = T.to_tensor(x).reshape(-1, 1, 1, name=name + "/reshape").wrapped
    res = T.dense(x, *inputs[1:], use_fc=True, name=name + "/gemm")
    if ndim == 2:
        res = res.reshape(-1, name=name)
    return [res.wrapped]
MX_TYPE_TO_BUILDER["FullyConnected"] = dense_nchw


class TRTClassificationModule(trtplus.Module):
    def __init__(self, weight_path, symbol_path, device=-1, shape=(1, 3, 224, 224)):
        super().__init__(device=device)
        self.weight_dict = create_weight_dict(weight_path)
        with open(symbol_path, "r") as f:
            self.symbol_json = json.load(f)
        
        self.shape = shape

    def build(self):
        inp = T.placeholder(trt.float32, self.shape,
                            name="data")  # must same as mxnet input name
        # Algorithm of mxnet2trt: start from leaf nodes,
        # recursive check input tensors until all input tensors are found in name_to_tensor.
        # tensors in inputs argument will be assigned to name_to_tensor before build.
        name_to_tensor, _ = mxnet2trt(
            T.get_trt_network(),
            self.symbol_json,
            {"data": [inp]
             },  # mxnet layer (node) name to outputs of that layer.
            self.weight_dict,
            leaf_layers=["softmax"],  # only build network before _plus1
        )
        return T.to_tensor(name_to_tensor["softmax"][0])

def load_checkpoint(weight_path):
    save_dict = mx.nd.load(weight_path)
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v
    return arg_params, aux_params

class MXClassificationModule:
    def __init__(self, weight_path, symbol_path, device=0, shape=(1,3,224,224)):
        self.arg_params, self.aux_params = load_checkpoint(weight_path)
        self.sym = mx.sym.load(symbol_path)
        self.shape = shape
        self.mod = mx.mod.Module(symbol=self.sym, context=mx.gpu(device), data_names=["data"])
        self.mod.bind(for_training=False, data_shapes=[("data", self.shape)])
        self.mod.set_params(self.arg_params, self.aux_params, allow_extra=False)

    def inference(self, np_data):
        mx_data = mx.io.DataBatch(data=[mx.nd.array(np_data)], label=None)
        self.mod.forward(mx_data)
        return self.mod.get_outputs()[0].asnumpy()


if __name__ == "__main__":
    from pycuda import driver 
    driver.init()
    model_dir = "model"
    weight_path = os.path.join(model_dir, "resnet18-0049.params")
    symbol_path = os.path.join(model_dir, "resnet18-symbol.json")
    calib_path = "resnet18_calib.json"
    shape = (3, 224, 224)
    fake_img = np.random.uniform(-1, 1, size=(1,3,224,224)).astype(np.float32)

    mod = TRTClassificationModule(weight_path, symbol_path, shape=shape, device=0)
    int8_calibrator = GeneratorEntropyCalibrator(input_names=["data"],
                                                 cache_path=calib_path)
    mod.build_engine(
        workspace=2**30,
        int8_calibrator=None,
        max_batch_size=1,
    )
    
    trt_res = mod(data=fake_img)

    mod = MXClassificationModule(weight_path, symbol_path, shape=(1,) + shape)
    mx_res = mod.inference(fake_img)

    assert trt_res.shape == mx_res.shape
    print(mx_res.shape)

    np.testing.assert_almost_equal(mx_res, trt_res, decimal=3)
    # print(mx_res - trt_res)
