import tensorflow as tf


def concat_relu(x):
    """ like concatenated ReLU (http://arxiv.org/abs/1603.05201), but then with ELU """
    axis = x.get_shape().ndims - 1
    return tf.nn.relu(tf.concat([x, -x], axis))