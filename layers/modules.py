import tensorflow as tf
from functools import reduce
from collections import namedtuple
from ops.convolutions import causal_conv_via_matmul, initial_X_prev
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
            effective_filter_width = self.filter_width + (self.filter_width - 1) * (self.dilation - 1)
            inputs = tf.pad(inputs, [[0, 0], [effective_filter_width - 1, 0], [0, 0]])
            inputs = tf.expand_dims(inputs, axis=1)
            Y = tf.nn.convolution(inputs, self.W, padding="VALID", dilation_rate=[1, self.dilation])
            Y = tf.squeeze(Y, axis=1)
            next_state = None
        if self.use_bias:
            Y += self.b
        Y = tf.nn.tanh(Y)
        return Y, next_state

    def zero_state(self, batch_size, in_channels, dtype):
        return initial_X_prev(batch_size, in_channels, self.filter_width, self.dilation, dtype)


class ResidualBlock(tf.layers.Layer):

    def __init__(self, residual_channels, filter_width, dilation,
                 use_filter_gate_bias, use_output_bias,
                 trainable=True, name=None, **kwargs):
        super(ResidualBlock, self).__init__(name=name, trainable=trainable, **kwargs)
        self.residual_channels = residual_channels
        self.filter_width = filter_width
        self.dilation = dilation
        self.use_filter_gate_bias = use_filter_gate_bias
        self.use_output_bias = use_output_bias

        self.gate_convolution = CausalConvolution(2 * self.residual_channels, filter_width, dilation=dilation,
                                                  use_bias=use_filter_gate_bias)

    def build(self, input_shape):
        self.lc_gc_filter_gate_W = self.add_variable("lc_gc_filter_gate_W",
                                                     [2 * self.residual_channels, 2 * self.residual_channels])

        self.output_W = self.add_variable("output_W", [self.residual_channels, self.residual_channels])
        if self.use_output_bias:
            self.output_b = self.add_variable("output_b", [self.residual_channels])

        self.built = True

    def call(self, inputs, state=None, sequential_inference_mode=False, output_width=None):
        block_input, condition = inputs
        Y, Y_state = self.gate_convolution(block_input, state=state,
                                           sequential_inference_mode=sequential_inference_mode)

        Y += tf.tensordot(condition, self.lc_gc_filter_gate_W, [[len(condition.shape.as_list()) - 1],
                                                                [0]])

        Y = tf.nn.tanh(Y[:, :, :self.residual_channels]) * tf.sigmoid(Y[:, :, self.residual_channels:])

        # skip signal
        skip_output = Y

        # output signal
        Z = tf.tensordot(Y, self.output_W, [[len(Y.shape.as_list()) - 1], [0]])  # 1x1 convolution
        if self.use_output_bias:
            Z += self.output_b

        # reconstruct output from input and residual signals.
        block_output = block_input + Z

        return block_output, skip_output, Y_state

    def zero_state(self, batch_size, in_channels, dtype):
        return initial_X_prev(batch_size, in_channels, self.filter_width, self.dilation, dtype)


class PostProcessing(tf.layers.Layer):

    def __init__(self, n_blocks, residual_channels, skip_channels, out_channels,
                 use_skip_bias, use_postprocessing1_bias, use_postprocessing2_bias,
                 trainable=True, name=None, **kwargs):
        super(PostProcessing, self).__init__(name=name, trainable=trainable, **kwargs)
        self.n_blocks = n_blocks
        self.skip_channels = skip_channels
        self.residual_channels = residual_channels
        self.out_channels = out_channels
        self.use_skip_bias = use_skip_bias
        self.use_postprocessing1_bias = use_postprocessing1_bias
        self.use_postprocessing2_bias = use_postprocessing2_bias

    def build(self, input_shape):
        # Perform concatenation -> 1x1 conv -> ReLU -> 1x1 conv -> ReLU -> 1x1 conv
        # Note that it is faster to perform concatenation of skip outputs and then a single matrix multiplication with weights than
        # a matrix multiplication of each skip output with the corresponding weights and then summations of the results.
        # skip connections
        self.W = self.add_variable("skip_W", [self.n_blocks * self.residual_channels, self.skip_channels])
        if self.use_skip_bias:
            self.skip_b = self.add_variable("skip_b", [self.skip_channels])

        self.postprocessing1_W = self.add_variable("postprocessing1_W", [2 * self.skip_channels, self.skip_channels])
        if self.use_postprocessing1_bias:
            self.postprocessing1_b = self.add_variable("postprocessing1_b", [self.skip_channels])

        self.postprocessing2_W = self.add_variable("postprocessing2_W", [2 * self.skip_channels, self.out_channels])
        if self.use_postprocessing2_bias:
            self.postprocessing2_b = self.add_variable("postprocessing2_b", [self.out_channels])

        self.built = True

    def call(self, skip_outputs, **kwargs):
        # Perform concatenation -> 1x1 conv -> ReLU -> 1x1 conv -> ReLU -> 1x1 conv
        assert len(skip_outputs) == self.n_blocks

        X_concat = tf.concat(skip_outputs, axis=-1)
        X = tf.tensordot(X_concat, self.W, [[len(X_concat.shape.as_list()) - 1], [0]])  # 1x1 convolution
        if self.use_skip_bias:
            X += self.skip_b

        X = concat_relu(X)

        X = tf.tensordot(X, self.postprocessing1_W, [[len(X.shape.as_list()) - 1], [0]])  # 1x1 convolution
        if self.use_postprocessing1_bias:
            X += self.postprocessing1_b

        X = concat_relu(X)

        probability_params = tf.tensordot(X, self.postprocessing2_W,
                                          [[len(X.shape.as_list()) - 1], [0]])  # 1x1 convolution
        if self.use_postprocessing2_bias:
            probability_params += self.postprocessing2_b

        return probability_params


class ConditionProjection(tf.layers.Layer):

    def __init__(self, residual_channels, local_condition_label_dim,
                 use_global_condition=False, global_condition_cardinality=None,
                 trainable=True, name=None, **kwargs):
        super(ConditionProjection, self).__init__(name=name, trainable=trainable, **kwargs)
        self.residual_channels = residual_channels
        self.local_condition_label_dim = local_condition_label_dim
        self.use_global_condition = use_global_condition
        self.global_condition_cardinality = global_condition_cardinality

    def build(self, _):
        self.W_lc = self.add_variable("W_lc", [self.local_condition_label_dim, 2 * self.residual_channels])
        if self.use_global_condition:
            self.W_gc = self.add_variable("W_gc", [self.global_condition_cardinality, 2 * self.residual_channels])
        self.built = True

    def call(self, local_condition, global_condition=None):
        H_lc = tf.tensordot(local_condition, self.W_lc, [[len(local_condition.shape.as_list()) - 1],
                                                         [0]])

        if self.use_global_condition:
            assert global_condition is not None
            H_gc = tf.nn.embedding_lookup(self.W_gc, global_condition)
            H = H_lc + H_gc
        else:
            H = H_lc
        return tf.nn.tanh(H)


class ProbabilityParameterEstimatorState(
    namedtuple("ProbabilityParameterEstimatorState", ["causal_layer_state", "residual_block_states"])):
    pass


class ProbabilityParameterEstimator(tf.layers.Layer):

    def __init__(self, filter_width, residual_channels, dilations,
                 skip_channels, out_channels,
                 use_causal_conv_bias, use_filter_gate_bias, use_output_bias,
                 use_skip_bias, use_postprocessing1_bias, use_postprocessing2_bias,
                 trainable=True, name=None, **kwargs):
        super(ProbabilityParameterEstimator, self).__init__(name=name, trainable=trainable, **kwargs)
        self.residual_channels = residual_channels
        self.filter_width = filter_width
        self.dilations = dilations
        self.out_channels = out_channels

        self.causal_layer = CausalConvolution(residual_channels, filter_width, dilation=1,
                                              use_bias=use_causal_conv_bias)

        j = self.filter_width - 1
        self.residual_blocks = []
        for dilation in dilations:
            j = j + dilation * (filter_width - 1)
            block = ResidualBlock(residual_channels, filter_width, dilation, use_filter_gate_bias,
                                  use_output_bias)
            self.residual_blocks.append(block)

        self.postprocessing = PostProcessing(len(dilations), residual_channels, skip_channels, out_channels,
                                             use_skip_bias, use_postprocessing1_bias, use_postprocessing2_bias)

    def build(self, _):
        self.built = True

    def call(self, inputs, state=None, sequential_inference_mode=False):
        # X -> causal_layer -> residual_block0 -> ... residual_blockl -> postprocessing -> P params
        state = ProbabilityParameterEstimatorState(None, [None] * len(self.dilations)) if state is None else state

        X, H = inputs

        input_width = tf.shape(X)[1]
        padding_width = self.filter_width - 1
        output_width = input_width + padding_width - self.receptive_field + 1 if not sequential_inference_mode else None

        # causal layer
        X, causal_layer_state = self.causal_layer(X, state=state.causal_layer_state,
                                                  sequential_inference_mode=sequential_inference_mode)

        # Residual blocks
        def residual_block(acc, dilation_layer_state):
            dilation, residual_block_layer, residual_block_state = dilation_layer_state
            j, X, skip_outputs, next_block_states = acc
            h = H if sequential_inference_mode else H
            X, skip_output, next_block_state = residual_block_layer((X, h), state=residual_block_state,
                                                                    sequential_inference_mode=sequential_inference_mode,
                                                                    output_width=output_width)
            return j, X, skip_outputs + [skip_output], next_block_states + [next_block_state]

        initial = (self.filter_width - 1, X, [], [])
        j, X, skip_outputs, block_states = reduce(residual_block, zip(self.dilations, self.residual_blocks,
                                                                      state.residual_block_states), initial)

        next_state = ProbabilityParameterEstimatorState(causal_layer_state, block_states)

        # Post-processing
        probability_params = self.postprocessing(skip_outputs)
        return probability_params, next_state

    def zero_state(self, batch_size, in_channels, dtype):
        state = ProbabilityParameterEstimatorState(self.causal_layer.zero_state(batch_size, in_channels, dtype),
                                                   [rb.zero_state(batch_size, self.residual_channels, dtype) for rb in
                                                    self.residual_blocks])
        return state

    @property
    def receptive_field(self):
        receptive_field = (self.filter_width - 1) * sum(self.dilations) + 1  # due to dilation layers
        receptive_field += self.filter_width - 1  # due to causal layer
        return receptive_field
