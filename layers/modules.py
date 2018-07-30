import tensorflow as tf
from functools import reduce
from collections import namedtuple
from ops.convolutions import causal_conv_via_matmul
from ops.ops import concat_relu


class CausalConvolution(tf.layers.Layer):

    def __init__(self, out_channels, filter_width, dilation, use_bias,
                 trainable=True, name=None, **kwargs):
        super(CausalConvolution, self).__init__(name=name, trainable=trainable, **kwargs)
        self.out_channels = out_channels
        self.filter_width = filter_width
        self.dilation = dilation
        self.use_bias = use_bias

    def build(self, input_shape):
        in_channels = input_shape[-1].value
        self.W = self.add_variable("W", [1, self.filter_width, in_channels, self.out_channels])
        if self.use_bias:
            self.b = self.add_variable("b", [self.out_channels])

        self.built = True

    def call(self, inputs, state=None, sequential_inference_mode=False):
        if sequential_inference_mode:
            Y, next_state = causal_conv_via_matmul(inputs, state, tf.squeeze(self.W, axis=0), dilation=self.dilation)
        else:
            if inputs.get_shape().ndims == 3:
                inputs = tf.expand_dims(inputs, axis=1)
            Y = tf.nn.convolution(inputs, self.W, padding="VALID", dilation_rate=[1, self.dilation])
            Y = tf.squeeze(Y, axis=1)
            next_state = None
        if self.use_bias:
            Y += self.b
        Y = tf.nn.tanh(Y)
        return Y, next_state

