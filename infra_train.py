import logging, os
import argparse
# import config
from config.edict_config import config
import mxnet as mx
from core.solver import Solver
from core.memonger_v2 import search_plan_to_layer
from core.callback import DetailSpeedometer
from data import *
from symbol import *
import datetime
import pprint
import horovod.mxnet as hvd

def main(config):
    # log file
    log_dir = os.path.join(config.output_dir, "./log")
    if not os.path.exists(log_dir):
        try:
            os.mkdir(log_dir)
        except Exception as e:
            print("Can't create directory: %s, maybe other worker created!" % log_dir)
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
    model_dir = os.path.join(config.output_dir, "./model")
    if not os.path.exists(model_dir):
        try:
            os.mkdir(model_dir)
        except Exception as e:
            print("Can't create directory: %s, maybe other worker created!" % model_dir)

    # set up environment
    if config.use_horovod == 1:
        print("--------------- using horovod to update parameters ---------------------")
        # Initialize Horovod
        hvd.init()
        devs = mx.cpu() if config.gpu_list is None or config.gpu_list == '-1' else mx.gpu(hvd.local_rank())
        num_workers = hvd.size()
        rank = hvd.rank()
        config.batch_size = config.batch_per_gpu
        # horovod divide num_workers implictly.
        rescale_grad = 1.0 / config.batch_size
    else:
        kv = mx.kvstore.create(config.kv_store)
        devs = mx.cpu() if config.gpu_list is None or config.gpu_list == '-1' else [mx.gpu(int(i)) for i in config.gpu_list]
        num_workers = kv.num_workers
        rank = kv.rank
        config.batch_size = config.batch_per_gpu * len(config.gpu_list)
        rescale_grad = 1.0 / config.batch_size / num_workers

    if config.network == 'test_symbol':
        config.batch_size = 10
        config.image_shape = (32, 32)
        config.num_classes = 10
    data_names = ('data',)
    label_names = ('softmax_label',)
    data_shapes = [('data', tuple([config.batch_size] + config.image_shape))]
    label_shapes = [('softmax_label', (config.batch_size,))]

    # set up iterator and symbol
    # iterator
    if config.network == 'test_symbol':
        train, val, num_examples = get_test_symbol_data(config.num_classes, config.batch_size, tuple(config.image_shape))
    else:
        train, val, num_examples = imagenet_iterator(data_dir=config.data_dir,
                                                     batch_size=config.batch_size,
                                                     num_workers=num_workers,
                                                     rank=rank,
                                                     image_shape=tuple(config.image_shape))
    if config.network == 'resnet' or config.network == 'resnet_v1':
        symbol = eval(config.network)(units=config.units,
                                      num_stage=config.num_stage,
                                      filter_list=config.filter_list,
                                      num_classes=config.num_classes,
                                      data_type=config.data_type,
                                      bottle_neck=config.bottle_neck,
                                      grad_scale=config.grad_scale,
                                      memonger=False)
    elif config.network == 'test_symbol':
        symbol = eval(config.network)(num_classes=config.num_classes)


    # mx.viz.print_summary(symbol, {'data': (1, 3, 224, 224)})

    # train
    epoch_size = max(int(num_examples / config.batch_size / num_workers), 1)
    if num_workers > 1:
        logging.info('Resizing training data to %d batches per machine', epoch_size)
        # resize train iter to ensure each machine has same number of batches per epoch
        # if not, dist_sync can hang at the end with one machine waiting for other machines
        train = mx.io.ResizeIter(train, epoch_size)

    lr_epoch = [int(epoch) for epoch in config.lr_step]
    lr_epoch_diff = [epoch - config.begin_epoch for epoch in lr_epoch if epoch > config.begin_epoch]
    lr = config.lr * (config.lr_factor ** (len(lr_epoch) - len(lr_epoch_diff)))
    lr_iters = [int(epoch * epoch_size) for epoch in lr_epoch_diff]
    print('using MultiFactorScheduler warmup lr', config.warmup_lr, 'warm_epoch', config.warm_epoch,
            'warm_step', int(config.warm_epoch * epoch_size), "lr: ", lr, "lr_epoch_diff: ",
            lr_epoch_diff, "lr_iters: ", lr_iters)
    lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(base_lr=lr, step=lr_iters, factor=config.lr_factor,
                                        warmup_mode='linear',
                                        warmup_begin_lr=config.warmup_lr,
                                        warmup_steps=int(config.warm_epoch * epoch_size))
    if (rank == 0):
        print("********* lr:{}, rescale_grad:{}, batch_size:{}, num_workers:{} **********".format(lr, rescale_grad, config.batch_size, num_workers))
    optimizer_params = {
        'learning_rate': lr,
        'wd': config.wd,
        'lr_scheduler': lr_scheduler,
        'rescale_grad': rescale_grad}
    # Only a limited number of optimizers have 'momentum' property
    has_momentum = {'sgd', 'dcasgd', 'nag', 'signum', 'lbsgd'}
    if config.optimizer in has_momentum:
        optimizer_params['momentum'] = config.momentum

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
    # epoch_end_callback = _save_model(os.path.join(model_dir, config.model_prefix), rank)
    batch_end_callback = mx.callback.Speedometer(config.batch_size, config.frequent)
    # batch_end_callback = DetailSpeedometer(config.batch_size, config.frequent)
    initializer = mx.init.Xavier(rnd_type='gaussian', factor_type='in', magnitude=2)
    arg_params = None
    aux_params = None
    if config.begin_epoch > 0:
        _, arg_params, aux_params = _load_model(os.path.join(model_dir, config.model_prefix), config.begin_epoch)

    if config.use_horovod == 1:
        opt = mx.optimizer.create(config.optimizer, sym=symbol, **optimizer_params)
        # opt = hvd.DistributedOptimizer(opt, rank)
        opt = hvd.DistributedOptimizer(opt)
        if arg_params is not None:
            hvd.broadcast_parameters(arg_params, root_rank=0)
        if aux_params is not None:
            hvd.broadcast_parameters(aux_params, root_rank=0)
        mx.nd.waitall()
        solver.fit(train_data=train,
                   eval_data=val,
                   eval_metric=eval_metric,
                   epoch_end_callback=epoch_end_callback,
                   batch_end_callback=batch_end_callback,
                   initializer=initializer,
                   arg_params=arg_params,
                   aux_params=aux_params,
                   optimizer=opt,
                   optimizer_params=optimizer_params,
                   begin_epoch=config.begin_epoch,
                   num_epoch=config.num_epoch,
                   kvstore=None,
                   rank=rank)
    else:
        solver.fit(train_data=train,
                   eval_data=val,
                   eval_metric=eval_metric,
                   epoch_end_callback=epoch_end_callback,
                   batch_end_callback=batch_end_callback,
                   initializer=initializer,
                   arg_params=arg_params,
                   aux_params=aux_params,
                   optimizer=config.optimizer,
                   optimizer_params=optimizer_params,
                   begin_epoch=config.begin_epoch,
                   num_epoch=config.num_epoch,
                   kvstore=kv,
                   rank=rank)

def _save_model(model_prefix, rank=0):
    if model_prefix is None:
        return None
    if rank != 0:
        return None
    return mx.callback.do_checkpoint(model_prefix if rank == 0 else "%s-%d" % (
        model_prefix, rank), period=1)

def _load_model(model_prefix, model_load_epoch):
    if model_load_epoch is None or model_load_epoch < 1:
        return (None, None, None)
    assert model_prefix is not None
    sym, arg_params, aux_params = mx.model.load_checkpoint(
        model_prefix, model_load_epoch)
    logging.info('Loaded model %s_%04d.params', model_prefix, model_load_epoch)
    return (sym, arg_params, aux_params)


if __name__ == '__main__':
    pprint.pprint(config)
    main(config)
