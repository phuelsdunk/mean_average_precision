from __future__ import absolute_import
from functools import partial

import tensorflow as tf

from tensorflow_mean_average_precision.python.ops import greedy_assignment_ops
from tensorflow.keras.metrics import Metric, AUC


@tf.function
def _batch_greedy_assignment(similarity_true_pred, threshold):
    return tf.map_fn(
        partial(greedy_assignment_ops.greedy_assignment, threshold=threshold),
        similarity_true_pred,
        fn_output_signature=tf.bool)


class MeanAveragePrecision(Metric):

    def __init__(self, thresholds, name=None):
        super(MeanAveragePrecision, self).__init__(name=name)

        self.thresholds = thresholds

        self.ap_metrics = [
            AUC(curve='pr', name='AveragePrecision @ %.3f' % threshold)
            for threshold in self.thresholds
        ]

    def reset_states(self):
        ops = [metric.reset_states() for metric in self.ap_metrics]
        return tf.group(ops, name='reset_states')

    def result(self):
        return tf.reduce_mean([metric.result() for metric in self.ap_metrics],
                              name='result')

    def update_state(self, similarity_true_pred, scores_pred,
                     sample_weight=None):
        # Sort predictions by scores
        index = tf.argsort(scores_pred,
                           axis=-1, direction='DESCENDING')
        similarity_true_pred = tf.gather(similarity_true_pred, index,
                                         axis=-1, batch_dims=1)
        scores_pred = tf.gather(scores_pred, index,
                                axis=-1, batch_dims=1)

        # Update all metrices
        ops = [metric.update_state(
            _batch_greedy_assignment(similarity_true_pred, threshold),
            scores_pred,
            sample_weight)
            for threshold, metric in zip(self.thresholds, self.ap_metrics)]
        return tf.group(ops, name='update_state')

    def get_config(self):
        return {'thresholds': self.thresholds}
