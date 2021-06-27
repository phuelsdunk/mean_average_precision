CXX := g++
NVCC := nvcc
PYTHON_BIN_PATH = python

GREEDY_ASSIGNMENT_SRCS = $(wildcard tensorflow_mean_average_precision/cc/kernels/*.cc) $(wildcard tensorflow_mean_average_precision/cc/ops/*.cc)

TF_CFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

CFLAGS = ${TF_CFLAGS} -fPIC -O2 -std=c++11
LDFLAGS = -shared ${TF_LFLAGS}

GREEDY_ASSIGNMENT_TARGET_LIB = tensorflow_mean_average_precision/python/ops/_greedy_assignment_ops.so

# greedy_assignment op for CPU
greedy_assignment_op: $(GREEDY_ASSIGNMENT_TARGET_LIB)

$(GREEDY_ASSIGNMENT_TARGET_LIB): $(GREEDY_ASSIGNMENT_SRCS)
	$(CXX) $(CFLAGS) -o $@ $^ ${LDFLAGS}

greedy_assignment_test: tensorflow_mean_average_precision/python/ops/greedy_assignment_ops_test.py tensorflow_mean_average_precision/python/ops/greedy_assignment_ops.py $(GREEDY_ASSIGNMENT_TARGET_LIB)
	$(PYTHON_BIN_PATH) tensorflow_mean_average_precision/python/ops/greedy_assignment_ops_test.py

mean_average_precision_pip_pkg: $(GREEDY_ASSIGNMENT_TARGET_LIB)
	./build_pip_pkg.sh make artifacts

clean:
	rm -f $(GREEDY_ASSIGNMENT_TARGET_LIB)
