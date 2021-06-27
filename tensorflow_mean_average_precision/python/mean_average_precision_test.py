from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.python.platform import test
from tensorflow_mean_average_precision.python.mean_average_precision \
    import MeanAveragePrecision


class TestMeanAveragePrecision(test.TestCase):

    def testTensorInput(self):
        targets = [
            [[0], [1], [2]],
            [[4], [5]]
        ]
        predictions = [
            [[2], [1], [0], [4], [5]],
            [[4], [5], [6], [7], [8]]
        ]
        scores = [
            [1.0, 0.8, 0.6, 0.0, 0.0],
            [1.0, 0.8, 0.0, 0.0, 0.0]
        ]

        targets = tf.ragged.constant(targets, tf.float32, ragged_rank=1)
        predictions = tf.constant(predictions, tf.float32)
        scores = tf.constant(scores, tf.float32)

        delta = targets[:, :, tf.newaxis] - predictions[:, tf.newaxis, :]
        similarity = tf.reduce_sum(tf.abs(delta), axis=-1)

        mean_ap = MeanAveragePrecision([0.5])
        mean_ap.update_state(similarity, scores)

        self.assertAllClose(mean_ap.result(), 1.0)


if __name__ == '__main__':
    test.main()
