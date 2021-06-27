from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.platform import test
try:
    from tensorflow_greedy_assignment.python.ops import .greedy_assignment_ops
except ImportError:
    from greedy_assignment_ops import greedy_assignment

if __name__ == '__main__':
    test.main()
