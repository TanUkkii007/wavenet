import tensorflow as tf
import numpy as np
from collections import namedtuple
from hypothesis import given, assume, settings, unlimited, HealthCheck
from hypothesis.strategies import integers, composite
from hypothesis.extra.numpy import arrays
from ops.convolutions import initial_X_prev
from layers.modules import CausalConvolution, ProbabilityParameterEstimator, ConditionProjection, ProbabilityParameterEstimatorState


class Args(namedtuple("Args", ["X", "batch_size", "n_classes", "filter_width", "width"])):
    pass

class ProbabilityParameterEstimatorArgs(namedtuple("ProbabilityParameterEstimatorArgs", ["X", "batch_size", "n_classes", "filter_width", "width", "dilations"])):
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

@composite
def probability_parameter_estimator_args(draw, batch_size=integers(1, 3), n_classes=integers(4, 10), filter_width=integers(2, 3),
                                         n_dilation=integers(1, 2)):
    batch_size = draw(batch_size)
    n_classes = draw(n_classes)
    filter_width = draw(filter_width)
    n_dilation = draw(n_dilation)
    dilations = [2 ** i for i in range(n_dilation)]
    receptive_field = filter_width - 1 + (filter_width - 1) * sum(dilations) + 1
    width = receptive_field * 5
    X = draw(arrays(dtype=np.float32, shape=[batch_size, width], elements=integers(0, n_classes - 1)))
    return ProbabilityParameterEstimatorArgs(X, batch_size, n_classes, filter_width, width, dilations)


class ModulesTest(tf.test.TestCase):

    @given(args=all_args(), out_channels=integers(3, 5), dilation=integers(1, 3))
    @settings(suppress_health_check=[HealthCheck.filter_too_much], timeout=unlimited)
    def test_causal_convolution(self, args, out_channels, dilation):
        X, batch_size, n_classes, filter_width, width = args
        assume(n_classes - filter_width - out_channels - (filter_width - 1) * (dilation - 1) + 1 > 0)
        X = tf.one_hot(X, n_classes)
        X_prev = initial_X_prev(batch_size, n_classes, filter_width, dilation, dtype=X.dtype)

        causal_convolution = CausalConvolution(out_channels, filter_width, use_bias=True, dilation=dilation)

        Y1 = []
        for t in range(width):
            y1, X_prev = causal_convolution(X[:, t, :], state=X_prev, sequential_inference_mode=True)
            Y1.append(y1)
        Y1 = tf.stack(Y1, axis=1)

        Y2, _ = causal_convolution(X)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            Y1, Y2 = sess.run([Y1, Y2])
            self.assertAllClose(Y2, Y1)

    @given(args=probability_parameter_estimator_args(), out_channels=integers(3, 5), skip_channels=integers(3, 5),
           local_condition_dim=integers(3, 5))
    @settings(max_examples=10, suppress_health_check=[HealthCheck.filter_too_much], timeout=unlimited)
    def test_probability_parameter_estimator(self, args, out_channels, skip_channels, local_condition_dim):
        X, batch_size, n_classes, filter_width, width, dilations = args
        # assume(n_classes - filter_width - out_channels - (filter_width - 1) * (dilations[-1] - 1) + 1 > 0)
        X = tf.one_hot(X, n_classes)
        width += filter_width - 1

        local_condition_dim = n_classes  # ToDo: remove
        condition_projection = ConditionProjection(n_classes, local_condition_dim)

        probability_parameter_estimator = ProbabilityParameterEstimator(filter_width, n_classes, dilations,
                                                                        skip_channels, out_channels,
                                                                        use_causal_conv_bias=True,
                                                                        use_filter_gate_bias=True, use_output_bias=True,
                                                                        use_skip_bias=True,
                                                                        use_postprocessing1_bias=True,
                                                                        use_postprocessing2_bias=True)


        H = condition_projection(X[:, 1:, :])
        H = tf.zeros_like(H) # ToDo: remove this line

        Y2, _ = probability_parameter_estimator((X[:, :-1, :], H))

        Y1 = []
        X_prev = probability_parameter_estimator.zero_state(batch_size, n_classes, tf.float32)
        for t in range(width - probability_parameter_estimator.receptive_field):
            X_t = tf.expand_dims(X[:, t, :], axis=1)
            H_t = tf.expand_dims(H[:, t+1, :], axis=1)
            y1, X_prev = probability_parameter_estimator((X_t, H_t), state=X_prev, sequential_inference_mode=True)
            Y1.append(y1)
        Y1 = tf.squeeze(tf.stack(Y1, axis=1), axis=2)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            Y1, Y2 = sess.run([Y1, Y2])
            self.assertAllClose(Y2, Y1)
