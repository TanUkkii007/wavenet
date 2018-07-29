import tensorflow as tf
import numpy as np
from collections import namedtuple
from hypothesis import given, assume, settings, unlimited, HealthCheck
from hypothesis.strategies import integers, composite
from hypothesis.extra.numpy import arrays
from ops.convolutions import causal_conv_via_matmul, initial_X_prev


class Args(namedtuple("Args", ["X", "batch_size", "n_classes", "filter_width", "width"])):
    pass


@composite
def all_args(draw, batch_size=integers(1, 3), n_classes=integers(4, 10), filter_width=integers(2, 3),
             width=integers(2, 10)):
    batch_size = draw(batch_size)
    n_classes = draw(n_classes)
    filter_width = draw(filter_width)
    width = filter_width + draw(width)
    X = draw(arrays(dtype=np.float32, shape=[batch_size, width], elements=integers(0, n_classes - 1)))
    return Args(X, batch_size, n_classes, filter_width, width)


class ConvolutionsTest(tf.test.TestCase):

    @given(args=all_args(), out_channels=integers(3, 5), dilation=integers(1, 2))
    @settings(suppress_health_check=[HealthCheck.filter_too_much], timeout=unlimited)
    def test_causal_conv_via_matmul(self, args, out_channels, dilation):
        X, batch_size, n_classes, filter_width, width = args
        assume(n_classes - filter_width - out_channels - (filter_width - 1) * (dilation - 1) + 1 > 0)
        X = tf.one_hot(X, n_classes)
        effective_filter_width = filter_width + (filter_width - 1) * (dilation - 1)
        X2 = tf.reshape(
            tf.pad(X, [[0, 0], [effective_filter_width - 1, 0], [0, 0]]),
            shape=[batch_size, 1, -1, n_classes])  # (B, height, width, in_channels)
        X_prev = initial_X_prev(batch_size, n_classes, filter_width, dilation, dtype=X.dtype)

        W = tf.random_normal([filter_width, n_classes, out_channels])
        W2 = tf.reshape(W, [1, filter_width, n_classes, out_channels])

        Y1 = []
        for t in range(width):
            y1, X_prev = causal_conv_via_matmul(X[:, t, :], X_prev, W, dilation)
            Y1.append(y1)
        Y1 = tf.stack(Y1, axis=1)

        Y2 = tf.nn.atrous_conv2d(X2, W2, rate=dilation, padding="VALID")
        Y2 = tf.squeeze(Y2, axis=1)

        with self.test_session() as sess:
            Y1, Y2 = sess.run([Y1, Y2])
            self.assertAllClose(Y2, Y1)
