import os
# import config
from config.edict_config import config
import mxnet as mx
from mxnet.io import DataBatch, DataIter
import numpy as np


def cifar10_iterator(data_dir, batch_size, kv, image_shape):
    num_examples = 50000

    train = mx.io.ImageRecordIter(
        path_imgrec=os.path.join(data_dir, "cifar10_train.rec"),
        label_width=1,
        data_name='data',
        label_name='softmax_label',
        data_shape=image_shape,
        batch_size=batch_size,
        pad=4,
        fill_value=127,  # only used when pad is valid
        rand_crop=True,
        max_random_scale=1.0,  # 480 with imagnet, 32 with cifar10
        min_random_scale=1.0,  # 256.0/480.0
        max_aspect_ratio=0,
        random_h=0,
        random_s=0,
        random_l=0,
        max_rotate_angle=0,
        max_shear_ratio=0,
        rand_mirror=True,
        shuffle=True,
        num_parts=kv.num_workers,
        part_index=kv.rank)

    val = mx.io.ImageRecordIter(
        path_imgrec=os.path.join(data_dir, "cifar10_val.rec"),
        label_width=1,
        data_name='data',
        label_name='softmax_label',
        batch_size=batch_size,
        data_shape=(3, 32, 32),
        rand_crop=False,
        rand_mirror=False,
        num_parts=kv.num_workers,
        part_index=kv.rank)

    return train, val, num_examples


def cifar100_iterator(data_dir, batch_size, kv, image_shape):
    num_examples = 50000

    train = mx.io.ImageRecordIter(
        path_imgrec=os.path.join(data_dir, "train.rec"),
        label_width=1,
        data_name='data',
        label_name='softmax_label',
        data_shape=image_shape,
        batch_size=batch_size,
        pad=4,
        fill_value=127,  # only used when pad is valid
        rand_crop=True,
        max_random_scale=1.0,  # 480 with imagnet, 32 with cifar10
        min_random_scale=1.0,  # 256.0/480.0
        max_aspect_ratio=0,
        random_h=0,
        random_s=0,
        random_l=0,
        max_rotate_angle=0,
        max_shear_ratio=0,
        rand_mirror=True,
        shuffle=True,
        num_parts=kv.num_workers,
        part_index=kv.rank)

    val = mx.io.ImageRecordIter(
        path_imgrec=os.path.join(data_dir, "test.rec"),
        label_width=1,
        data_name='data',
        label_name='softmax_label',
        batch_size=batch_size,
        data_shape=(3, 32, 32),
        rand_crop=False,
        rand_mirror=False,
        num_parts=kv.num_workers,
        part_index=kv.rank)

    return train, val, num_examples

