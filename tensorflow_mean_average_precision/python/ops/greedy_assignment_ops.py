from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

greedy_assignment_ops = load_library.load_op_library(
    resource_loader.get_path_to_datafile('_greedy_assignment_ops.so'))
greedy_assignment = greedy_assignment_ops.greedy_assignment
