from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.platform import test
try:
    from tensorflow_mean_average_precision.python.ops \
        import greedy_assignment_ops
except ImportError:
    import greedy_assignment_ops


class TestGreedyAssignment(test.TestCase):

    def testThatPredictionsArePrioritized(self):
        with self.test_session():
            similarity = np.ones((3, 6))
            assignment = greedy_assignment_ops.greedy_assignment(similarity, 0)
            self.assertAllEqual(assignment,
                                [True, True, True, False, False, False])

    def testThatThresholdIsReached(self):
        with self.test_session():
            similarity = np.zeros((3, 6))
            assignment = greedy_assignment_ops.greedy_assignment(similarity, 0)
            self.assertAllEqual(assignment,
                                [False, False, False, False, False, False])

    def testThatBadMatchesAreDiscarded(self):
        with self.test_session():
            similarity = np.array(
                [[1, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0]])
            assignment = greedy_assignment_ops.greedy_assignment(similarity, 0)
            self.assertAllEqual(assignment,
                                [True, False, True, False, False])

    def testThatMatchedTargetIsIgnored(self):
        with self.test_session():
            similarity = np.array(
                [[1, 0, 0, 2, 1],
                 [2, 3, 4, 0, 1]])
            assignment = greedy_assignment_ops.greedy_assignment(similarity, 0)
            self.assertAllEqual(assignment,
                                [True, False, False, True, False])


if __name__ == '__main__':
    test.main()
