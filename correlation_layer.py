import tensorflow as tf
correlation_module = tf.load_op_library("./build/libcorrelation.so")

#Import and register the correltion gradient function
import _correlation_grad

corr = correlation_module.correlation
