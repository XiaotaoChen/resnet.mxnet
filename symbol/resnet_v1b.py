import mxnet as mx
from mxnext.complicate import normalizer_factory
from mxnext.simple import reluconv, conv, pool, relu, add, whiten, var, to_fp16

def resnet_unit(data, name, filter, stride, dilate, proj, norm_type, norm_mom, ndev):
    """
    One resnet unit is comprised of 2 or 3 convolutions and a shortcut.
    :param data:
    :param name:
    :param filter:
    :param stride:
    :param dilate:
    :param proj:
    :param norm_type:
    :param norm_mom:
    :param ndev:
    :return:
    """
    norm = normalizer_factory(type=norm_type, ndev=ndev, mom=norm_mom)

    conv1 = conv(data, name=name + "_conv1", filter=filter // 4)
    bn1 = norm(data=conv1, name=name + "_bn1")
    relu1 = relu(bn1, name=name + "_relu1")

    conv2 = conv(relu1, name=name + "_conv2", filter=filter // 4, kernel=3, stride=stride, dilate=dilate)
    bn2 = norm(data=conv2, name=name + "_bn2")
    relu2 = relu(bn2, name=name + "_relu2")

    conv3 = conv(relu2, name=name + "_conv3", filter=filter)
    bn3 = norm(data=conv3, name=name + "_bn3")

    if proj:
        shortcut = conv(data, name=name + "_sc", filter=filter, stride=stride)
        shortcut = norm(data=shortcut, name=name + "_sc_bn")
    else:
        shortcut = data

    eltwise = add(bn3, shortcut, name=name + "_plus")

    return relu(eltwise, name=name + "_relu")

def resnet_stage(data, name, num_block, filter, stride, dilate, norm_type, norm_mom, ndev):
    """
    One resnet stage is comprised of multiple resnet units. Refer to depth config for more information.
    :param data:
    :param name:
    :param num_block:
    :param filter:
    :param stride:
    :param dilate:
    :param norm_type:
    :param norm_mom:
    :param ndev:
    :return:
    """
    s, d = stride, dilate

    data = resnet_unit(data, "{}_unit1".format(name), filter, s, d, True, norm_type, norm_mom, ndev)
    for i in range(2, num_block + 1):
        data = resnet_unit(data, "{}_unit{}".format(name, i), filter, 1, d, False, norm_type, norm_mom, ndev)

    return data

def resnet_c1(data, use_3x3_conv0, use_bn_preprocess, norm_type, norm_mom, ndev):
    """
    Resnet C1 is comprised of irregular initial layers.
    :param data: image symbol
    :param use_3x3_conv0: use three 3x3 convs to replace one 7x7 conv
    :param use_bn_preprocess: use batchnorm as the whitening layer, introduced by tornadomeet
    :param norm_type: normalization method of activation, could be local, fix, sync, gn, in, ibn
    :param norm_mom: normalization momentum, specific to batchnorm
    :param ndev: num of gpus for sync batchnorm
    :return: C1 symbol
    """
    # preprocess
    if use_bn_preprocess:
        data = whiten(data, name="bn_data")

    norm = normalizer_factory(type=norm_type, ndev=ndev, mom=norm_mom)

    # C1
    if use_3x3_conv0:
        data = conv(data, filter=64, kernel=3, stride=2, name="conv0_0")
        data = norm(data, name='bn0_0')
        data = relu(data, name='relu0_0')

        data = conv(data, filter=64, kernel=3, name="conv0_1")
        data = norm(data, name='bn0_1')
        data = relu(data, name='relu0_1')

        data = conv(data, filter=64, kernel=3, name="conv0_2")
        data = norm(data, name='bn0_2')
        data = relu(data, name='relu0_2')
    else:
        data = conv(data, filter=64, kernel=7, stride=2, name="conv0")
        data = norm(data, name='bn0')
        data = relu(data, name='relu0')

    data = pool(data, name="pool0", kernel=3, stride=2, pool_type='max')

    return data

def resnet_factory(depth, use_3x3_conv0, use_bn_preprocess, norm_type="local", norm_mom=0.9, ndev=None, fp16=False):
    depth_config = {
            50: (3, 4, 6, 3),
            101: (3, 4, 23, 3),
            152: (3, 8, 36, 3),
            200: (3, 24, 36, 3)
    }
    num_c2_unit, num_c3_unit, num_c4_unit, num_c5_unit = depth_config[depth]

    data = var("data")
    if fp16:
        data = to_fp16(data, "data_fp16")
    c1 = resnet_c1(data, use_3x3_conv0, use_bn_preprocess, norm_type, norm_mom, ndev)
    c2 = resnet_stage(c1, "stage1", num_c2_unit, 256, 1, 1, norm_type, norm_mom, ndev)
    c3 = resnet_stage(c2, "stage2", num_c3_unit, 512, 2, 1, norm_type, norm_mom, ndev)
    c4 = resnet_stage(c3, "stage3", num_c4_unit, 1024, 2, 1, norm_type, norm_mom, ndev)
    c5 = resnet_stage(c4, "stage4", num_c5_unit, 2048, 2, 1, norm_type, norm_mom, ndev)


    return c5

def resnet50_v1b(num_classes):
    c5 = resnet_factory(50, False, False)
    pool1 = mx.symbol.Pooling(data=c5, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')
    flat = mx.symbol.Flatten(data=pool1)
    fc1 = mx.symbol.FullyConnected(data=flat, num_hidden=num_classes, name='fc1')
    cls = mx.symbol.SoftmaxOutput(data=fc1, name='softmax')
    return cls


