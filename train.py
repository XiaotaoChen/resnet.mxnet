import logging, os
import argparse
# import config
from config.edict_config import config
import mxnet as mx
from core.scheduler import multi_factor_scheduler
from core.solver import Solver
from core.memonger_v2 import search_plan_to_layer
from core.callback import DetailSpeedometer
from data import *
from symbol import *
import datetime
import pprint
from core.scheduler import WarmupMultiFactorScheduler

def main(config):
    output_dir = "experiments/" + config.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)s %(levelname)s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename='{}/{}.log'.format(output_dir, config.model_prefix),
                        filemode='a')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)s %(levelname)s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    logging.info(config)

    # set up environment
    devs = [mx.gpu(int(i)) for i in config.gpu_list]
    kv = mx.kvstore.create(config.kv_store)

    # set up iterator and symbol
    # iterator
    if config.use_multiple_iter is True:
        train, val, num_examples = multiple_imagenet_iterator(data_dir=config.data_dir,
                                                    batch_size=config.batch_size,
                                                    num_parts=2,
                                                    image_shape=tuple(config.image_shape),
                                                    data_nthread=config.data_nthreads)
    # elif config.use_dali_iter is True:
    #     train, val, num_examples = get_dali_iter(data_dir=config.data_dir,
    #                                              batch_size=config.batch_size,
    #                                              kv=kv,
    #                                              image_shape=tuple(config.image_shape),
    #                                              num_gpus=len(devs))
    else:
        if config.dataset == 'imagenet':
            train, val, num_examples = imagenet_iterator(data_dir=config.data_dir,
                                                         batch_size=config.batch_size,
                                                         kv=kv,
                                                         image_shape=tuple(config.image_shape))
        elif config.dataset == 'cifar10':
            train, val, num_examples = cifar10_iterator(data_dir=config.data_dir,
                                                         batch_size=config.batch_size,
                                                         kv=kv,
                                                         image_shape=tuple(config.image_shape))
        elif config.dataset == 'cifar100':
            train, val, num_examples = cifar100_iterator(data_dir=config.data_dir,
                                                         batch_size=config.batch_size,
                                                         kv=kv,
                                                         image_shape=tuple(config.image_shape))
    logging.info(train)
    logging.info(val)

    data_names = ('data',)
    label_names = ('softmax_label',)
    data_shapes = [('data', tuple([config.batch_size] + config.image_shape))]
    label_shapes = [('softmax_label', (config.batch_size,))]

    if config.network in ['resnet', 'resnet_cifar10', 'resnet_imagenet']:
        symbol = eval(config.network)(units=config.units,
                                      num_stage=config.num_stage,
                                      filter_list=config.filter_list,
                                      num_classes=config.num_classes,
                                      data_type=config.data_type,
                                      bottle_neck=config.bottle_neck,
                                      grad_scale=config.grad_scale,
                                      memonger=config.memonger,
                                      dataset_type=config.dataset)
    elif config.network == 'resnet_mxnet':
        symbol = eval(config.network)(units=config.units,
                                      num_stage=config.num_stage,
                                      filter_list=config.filter_list,
                                      num_classes=config.num_classes,
                                      data_type=config.data_type,
                                      bottle_neck=config.bottle_neck,
                                      grad_scale=config.grad_scale)
    elif config.network == 'resnext' or config.network == 'resnext_cyt':
        symbol = eval(config.network)(units=config.units,
                                      num_stage=config.num_stage,
                                      filter_list=config.filter_list,
                                      num_classes=config.num_classes,
                                      data_type=config.data_type,
                                      num_group=config.num_group,
                                      bottle_neck=config.bottle_neck)
    elif config.network == 'vgg16' or config.network == 'mobilenet' or config.network == 'shufflenet':
        symbol = eval(config.network)(num_classes=config.num_classes)
    elif config.network == "cifar10_sym":
        symbol = eval(config.network)()


    if config.fix_bn:
        from core.graph_optimize import fix_bn
        print("********************* fix bn ***********************")
        symbol = fix_bn(symbol)

    if config.quantize_flag:
        assert config.data_type == "float32", "current quantization op only support fp32 mode."
        from core.graph_optimize import attach_quantize_node
        worker_data_shape = dict(data_shapes + label_shapes)
        _, out_shape, _ = symbol.get_internals().infer_shape(**worker_data_shape)
        out_shape_dictoinary = dict(zip(symbol.get_internals().list_outputs(), out_shape))
        symbol = attach_quantize_node(symbol, out_shape_dictoinary, config.quantize_op_name, 
                                      config.quant_attrs["weight_quant_attrs"], config.quant_attrs["act_quant_attrs"], 
                                      config.quantized_op, config.skip_quantize_counts)
        # symbol.save("attach_quant.json")
        # raise NotImplementedError

    # symbol.save(config.network + ".json")
    # raise NotImplementedError
    # mx.viz.print_summary(symbol, {'data': (1, 3, 224, 224)})

    # memonger
    if config.memonger:
        # infer shape
        data_shape_dict = dict(train.provide_data + train.provide_label)
        per_gpu_data_shape_dict = {}
        for key in data_shape_dict:
            per_gpu_data_shape_dict[key] = (config.batch_per_gpu,) + data_shape_dict[key][1:]

        # if config.network == 'resnet':
        #     last_block = 'conv3_1_relu'
        #     if kv.rank == 0:
        #         logging.info("resnet do memonger up to {}".format(last_block))
        # else:
        #     last_block = None
        last_block = 'stage4_unit1_sc'
        input_dtype = {k: 'float32' for k in per_gpu_data_shape_dict}
        symbol = search_plan_to_layer(symbol, last_block, 1000, type_dict=input_dtype, **per_gpu_data_shape_dict)

    # train
    epoch_size = max(int(num_examples / config.batch_size / kv.num_workers), 1)
    if 'dist' in config.kv_store and not 'async' in config.kv_store \
            and config.use_multiple_iter is False and config.use_dali_iter is False:
        logging.info('Resizing training data to %d batches per machine {}'.format(epoch_size))
        # resize train iter to ensure each machine has same number of batches per epoch
        # if not, dist_sync can hang at the end with one machine waiting for other machines
        train = mx.io.ResizeIter(train, epoch_size)

    if config.warmup is not None and config.warmup is True:
        lr_epoch = [int(epoch) for epoch in config.lr_step]
        lr_epoch_diff = [epoch - config.begin_epoch for epoch in lr_epoch if epoch > config.begin_epoch]
        lr = config.lr * (config.lr_factor ** (len(lr_epoch) - len(lr_epoch_diff)))
        lr_iters = [int(epoch * epoch_size) for epoch in lr_epoch_diff]
        logging.info('warmup lr:{}, warm_epoch:{}, warm_step:{}, '.format(
            config.warmup_lr, config.warm_epoch, int(config.warm_epoch * epoch_size)))
        if config.lr_scheduler == 'poly':
            logging.info('PolyScheduler lr'.format(lr))
            lr_scheduler = mx.lr_scheduler.PolyScheduler(int(epoch_size*config.num_epoch), base_lr=lr, pwr=2, final_lr=0,
                                                         warmup_steps=int(config.warm_epoch * epoch_size),
                                                         warmup_begin_lr=0, warmup_mode='linear')
        else:
            logging.info('WarmupMultiFactorScheduler lr:{}, epoch size:{}, lr_epoch_diff:{}, '
                         'lr_iters:{}'.format( lr, epoch_size, lr_epoch_diff, lr_iters))
            lr_scheduler = WarmupMultiFactorScheduler(base_lr=lr, step=lr_iters, factor=config.lr_factor,
                                                  warmup=True, warmup_type='gradual',
                                                  warmup_lr=config.warmup_lr, warmup_step=int(config.warm_epoch * epoch_size))
    elif config.lr_step is not None:
        lr_epoch_diff = [epoch - config.begin_epoch for epoch in config.lr_step if epoch > config.begin_epoch]
        lr = config.lr * (config.lr_factor **(len(config.lr_step) - len(lr_epoch_diff)))
        lr_scheduler = multi_factor_scheduler(config.begin_epoch, epoch_size, step=config.lr_step,
                                              factor=config.lr_factor)
        step_ = [epoch * epoch_size for epoch in lr_epoch_diff]
        logging.info('multi_factor_scheduler lr:{}, epoch size:{}, epoch diff:{}, '
                     'step:{}'.format(lr, epoch_size, lr_epoch_diff, step_))
    else:
        lr = config.lr
        lr_scheduler = None
    print("begin epoch:{}, num epoch:{}".format(config.begin_epoch, config.num_epoch))

    optimizer_params = {
        'learning_rate': lr,
        'wd': config.wd,
        'lr_scheduler': lr_scheduler,
        'multi_precision': config.multi_precision}
    # Only a limited number of optimizers have 'momentum' property
    has_momentum = {'sgd', 'dcasgd', 'nag', 'signum', 'lbsgd'}
    if config.optimizer in has_momentum:
        optimizer_params['momentum'] = config.momentum
    # A limited number of optimizers have a warmup period
    has_warmup = {'lbsgd', 'lbnag'}
    if config.optimizer in has_warmup:
        optimizer_params['updates_per_epoch'] = epoch_size
        optimizer_params['begin_epoch'] = config.begin_epoch
        optimizer_params['batch_scale'] = 1.0
        optimizer_params['warmup_strategy'] = 'lars'
        optimizer_params['warmup_epochs'] = config.warm_epoch # not work whne warmup_strategy is 'lars'
        optimizer_params['num_epochs'] = config.num_epoch

    eval_metric = ['acc']
    if config.dataset == "imagenet":
        eval_metric.append(mx.metric.create('top_k_accuracy', top_k=5))

    solver = Solver(symbol=symbol,
                    data_names=data_names,
                    label_names=label_names,
                    data_shapes=data_shapes,
                    label_shapes=label_shapes,
                    logger=logging,
                    context=devs,
                    # for evaluate fold bn
                    config=config)
    epoch_end_callback = mx.callback.do_checkpoint(os.path.join(output_dir, config.model_prefix))
    batch_end_callback = mx.callback.Speedometer(config.batch_size, config.frequent)
    # batch_end_callback = DetailSpeedometer(config.batch_size, config.frequent)
    initializer = mx.init.Xavier(rnd_type='gaussian', factor_type='in', magnitude=2)
    arg_params = None
    aux_params = None
    if config.retrain:
        print('******************** retrain load pretrain model from: {}'.format(config.model_load_prefix))
        _, arg_params, aux_params = mx.model.load_checkpoint("{}".format(config.model_load_prefix),
                                                             config.model_load_epoch)
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
               allow_missing=config.allow_missing)

if __name__ == '__main__':
    # args = parse_args()
    now = datetime.datetime.now()
    date = '{}_{:0>2}_{:0>2}'.format(now.year, now.month, now.day)

    # set_config(args)
    main(config)
