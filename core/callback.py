import time
import logging
from collections import namedtuple

BatchEndParam = namedtuple('BatchEndParams',
                          ['epoch',
                           'nbatch',
                           'eval_metric',
                           'locals',
                           'rank',
                           'total_iter',
                           'cur_data_time',
                           'avg_data_time',
                           'cur_forward_time',
                           'avg_forward_time',
                           'cur_backward_time',
                           'avg_backward_time',
                           'cur_kvstore_sync_time',
                           'avg_kvstore_sync_time',
                           'cur_iter_total_time',
                           'avg_iter_total_time'
                           ])

class DetailSpeedometer(object):
    """Logs training speed and evaluation metrics periodically.

    Parameters
    ----------
    batch_size: int
        Batch size of data.
    frequent: int
        Specifies how frequently training speed and evaluation metrics
        must be logged. Default behavior is to log once every 50 batches.
    auto_reset : bool
        Reset the evaluation metrics after each log.

    Example
    -------
    >>> # Print training speed and evaluation metrics every ten batches. Batch size is one.
    >>> module.fit(iterator, num_epoch=n_epoch,
    ... batch_end_callback=mx.callback.Speedometer(1, 10))
    """
    def __init__(self, batch_size, frequent=50, auto_reset=True):
        self.batch_size = batch_size
        self.frequent = frequent
        self.init = False
        self.tic = 0
        self.last_count = 0
        self.auto_reset = auto_reset

    def __call__(self, param):
        """Callback to Show speed."""
        count = param.nbatch
        rank = param.rank
        total_iter = param.total_iter
        if self.last_count > count:
            self.init = False
        self.last_count = count

        if self.init:
            if count % self.frequent == 0:
                speed = self.frequent * self.batch_size / (time.time() - self.tic)
                if param.eval_metric is not None:
                    name_value = param.eval_metric.get_name_value()
                    if self.auto_reset:
                        param.eval_metric.reset()
                    # msg = 'Epoch[%d] Batch [%d]\tSpeed: %.2f samples/sec'
                    # msg += '\t%s=%f'*len(name_value)
                    # logging.info(msg, param.epoch, count, speed, *sum(name_value, ()))
                    msg = 'Epoch[%d] Rank[%d] Batch[%d] TotalIter[%d] forward:%.3f(%.3f)\tbackward:%.3f(%.3f)\tkv_sync:%.3f(%.3f)\tdata:%.3f(%.3f)\titer_total_time:%.3f(%.3f)\tSpeed: %.2f samples/sec'
                    msg += '\t%s=%f' * len(name_value)
                    logging.info(msg, param.epoch, rank, count, total_iter,
                                 param.cur_forward_time, param.avg_forward_time,
                                 param.cur_backward_time, param.avg_backward_time,
                                 param.cur_kvstore_sync_time, param.avg_kvstore_sync_time,
                                 param.cur_data_time, param.avg_data_time,
                                 param.cur_iter_total_time, param.avg_iter_total_time,
                                 speed, *sum(name_value, ()))
                else:
                    # logging.info("Iter[%d] Batch [%d]\tSpeed: %.2f samples/sec",
                    #              param.epoch, count, speed)
                    logging.info(
                        "Iter[%d] Rank[%d] Batch[%d] TotalIter[%d] forward:%.3f(%.3f)\tbackward:%.3f(%.3f)\tkv_sync:%.3f(%.3f)\tdata:%.3f(%.3f)\titer_total_time:%.3f(%.3f)\tSpeed: %.2f samples/sec",
                        param.epoch, rank, count, total_iter,
                        param.cur_forward_time, param.avg_forward_time,
                        param.cur_backward_time, param.avg_backward_time,
                        param.cur_kvstore_sync_time, param.avg_kvstore_sync_time,
                        param.cur_data_time, param.avg_data_time,
                        param.cur_iter_total_time, param.avg_iter_total_time,
                        speed)
                self.tic = time.time()
        else:
            self.init = True
            self.tic = time.time()