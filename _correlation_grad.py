#!/usr/bin/env python3
"""
Gradients for inner product.
"""
 
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops
correlation_grad_module = tf.load_op_library('./build/libcorrelation_grad.so')
 
@ops.RegisterGradient("Correlation")
def _correlation_grad_cc(op, grad):
    """
    The gradient for `correlation` using the operation implemented in C++.
    
    :param op: `correlation` `Operation` that we are differentiating, which we can use
        to find the inputs and outputs of the original op.
    :param grad: gradient with respect to the output of the `correlation` op.
    :return: gradients with respect to the input of `correlation`.
    """
    
    return correlation_grad_module.correlation_grad(grad, op.inputs[0], op.inputs[1],stride=op.get_attr('stride'),max_displacement=op.get_attr('max_displacement'))
