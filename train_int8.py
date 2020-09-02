import logging, os
import sys

import config

sys.path.insert(0, config.mxnet_path)
import mxnet as mx
from core.scheduler import multi_factor_scheduler
from core.solver import Solver
from data import *
from symbol import *

from pipeline_api import attach_quantize_node
from pipeline_api import SineScheduler


def main(config):
    # log file
    log_dir = "./log"
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)s %(levelname)s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename='{}/{}.log'.format(log_dir, config.model_prefix),
                        filemode='a')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)s %(levelname)s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    # model folder
    model_dir = "./model"
    # quantized model dir
    quantize_str = "w_{}{}_act_{}{}_{}e_{}e".format(config.weight_setting["quantize_op_name"],
                                                config.weight_setting["attrs"]["nbits"],
                                                config.act_setting["quantize_op_name"],
                                                config.act_setting["attrs"]["nbits"],
                                                config.quant_begin_epoch, config.quant_end_epoch)
    model_dir = os.path.join(model_dir, quantize_str)

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    # set up environment
    devs = [mx.gpu(int(i)) for i in config.gpu_list]
    kv = mx.kvstore.create(config.kv_store)

    # set up iterator and symbol
    # iterator
    train, val, num_examples = imagenet_iterator(data_dir=config.data_dir,
                                                 batch_size=config.batch_size,
                                                 kv=kv)
    print(train)
    print(val)
    data_names = ('data',)
    label_names = ('softmax_label',)
    data_shapes = [('data', (config.batch_size, 3, 224, 224))]
    label_shapes = [('softmax_label', (config.batch_size,))]

    # attach quantized node
    sym, arg_params, aux_params = mx.model.load_checkpoint("model/{}".format(config.model_load_prefix),
                                                             config.model_load_epoch)
    worker_data_shape = {"data":(1, 3, 224, 224)}
    _, out_shape, _ = sym.get_internals().infer_shape(**worker_data_shape)
    out_shape_dictoinary = dict(zip(sym.get_internals().list_outputs(), out_shape))

    sym = attach_quantize_node(sym, out_shape_dictoinary, config.weight_setting, config.act_setting, 
                               quantized_op=config.quantized_op, skip_quantize_counts=config.skip_quantize_counts,
                               quantize_counts=config.quantize_counts,)
    # sym.save("quant_sym.json")
    # raise NotImplementedError


    epoch_size = max(int(num_examples / config.batch_size / kv.num_workers), 1)
    # int8 training 
    base_lr = config.quant_lr
    lr_scheduler = SineScheduler(config.quant_lr, config.quant_epoch * epoch_size)

    optimizer_params = {'learning_rate': base_lr,
                        'lr_scheduler': lr_scheduler,
                        'wd': config.wd,
                        'momentum': config.momentum}
    # optimizer = "nag"
    optimizer = 'sgd'
    eval_metric = ['acc']
    if config.dataset == "imagenet":
        eval_metric.append(mx.metric.create('top_k_accuracy', top_k=5))

    solver = Solver(symbol=symbol,
                    data_names=data_names,
                    label_names=label_names,
                    data_shapes=data_shapes,
                    label_shapes=label_shapes,
                    logger=logging,
                    context=devs)
    epoch_end_callback = mx.callback.do_checkpoint(os.path.join(model_dir, config.model_prefix))
    batch_end_callback = mx.callback.Speedometer(config.batch_size, config.frequent)
    initializer = mx.init.Xavier(rnd_type='gaussian', factor_type='in', magnitude=2)

    solver.fit(train_data=train,
               eval_data=val,
               eval_metric=eval_metric,
               epoch_end_callback=epoch_end_callback,
               batch_end_callback=batch_end_callback,
               initializer=initializer,
               arg_params=arg_params,
               aux_params=aux_params,
               optimizer=optimizer,
               optimizer_params=optimizer_params,
               begin_epoch=config.quant_begin_epoch,
               num_epoch=config.quant_end_epoch,
               kvstore=kv)


if __name__ == '__main__':
    main(config)
