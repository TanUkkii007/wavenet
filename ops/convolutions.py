import tensorflow as tf


def causal_conv_via_matmul(X_cur, X_prev, W, dilation=1):
    '''
    :param X_cur: (B, in_channels)
    :param X_prev: (B, filter_width, in_channels)
    :param W: (in_channels, filter_width, out_channels)
    :param dilation:
    :return: # (B, out_channels)
    '''
    W_shape = tf.shape(W)
    batch_size = tf.shape(X_cur)[0]
    out_channels = W_shape[2]
    W = tf.reshape(W, shape=[-1, out_channels])  # (in_channels*filter_width, out_channels)
    # shift X by one
    X_next = tf.concat([tf.slice(X_prev, begin=[0, 1, 0], size=[-1, -1, -1]),
                        tf.expand_dims(X_cur, axis=1)],
                       axis=1)
    inputs = tf.reshape(X_next[:, 0::dilation, :], shape=[batch_size, -1])  # (B, filter_width*in_channels)
    conv_output = tf.matmul(inputs, W)  # (B, out_channels)
    return conv_output, X_next


def initial_X_prev(batch_size, in_channels, filter_width, dilation, dtype):
    effective_filter_width = filter_width + (filter_width - 1) * (dilation - 1)
    return tf.zeros([batch_size, effective_filter_width, in_channels], dtype)
