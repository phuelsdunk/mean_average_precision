from __future__ import absolute_import
from functools import partial

from tensorflow_mean_average_precision.python.ops import greedy_assignment_ops
from tensorflow.keras.metrics import Metric, AUC


class MeanAveragePrecision(Metric):

    def __init__(self, similarity_fn, thresholds, name=None):
        super(MeanAveragePrecision, self).__init__(name=name)

        self.similarity_fn = similarity_fn
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

    def update_state(self, y_true, y_pred, sample_weight=None):
        true_locations, true_scores = y_true
        pred_locations, pred_scores = y_pred
        
        similarities = self.similarity_fn(true_locations, pred_locations)

        batch_dims = 1
        true_slice = (
            batch_dims * (slice(None), )
            + (len(true_locations.shape) - batch_dims - 2) * (slice(None), )
            + (len(pred_locations.shape) - batch_dims - 2) * (tf.newaxis, )
        )
        similarities_weights = true_scores[true_slice]

        instance_similarities = tf.math.divide_no_nan(
            tf.reduce_sum(similarities * similarities_weights, axis=-1),
            tf.reduce_sum(similarities_weights, axis=-1))

        pred_match_masks = [
            tf.map_fn(
                partial(
                    greedy_assignment_ops.greedy_assignment,
                    threshold=threshold),
                instance_similarities)
            for threshold in self.thresholds
        ]
        pred_match_scores = tf.reduce_mean(pred_scores, axis=-1)

        ops = [
            metric.update_state(
            pred_match_mask,
            pred_match_scores,
            sample_weight)
            for pred_match_mask, metric
            in zip(pred_match_masks, self.ap_metrics)
        ]

        return tf.group(ops, name='update_state')

    def get_config(self):
        return {'thresholds': self.thresholds,
                'sigmas': self.sigmas}

