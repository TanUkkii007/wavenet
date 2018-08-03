import tensorflow as tf
from layers.modules import ProbabilityParameterEstimator, ConditionProjection
from ops.mixture_of_logistics_distribution import discretized_mix_logistic_loss
from ops.optimizers import get_learning_rate


class WaveNetModel(tf.estimator.Estimator):

    def __init__(self, params, model_dir=None, config=None, warm_start_from=None):
        def model_fn(features, labels, mode, params):
            is_training = mode == tf.estimator.ModeKeys.TRAIN
            is_validation = mode == tf.estimator.ModeKeys.EVAL
            is_prediction = mode == tf.estimator.ModeKeys.PREDICT

            condition_projection = ConditionProjection(params.residual_channels,
                                                       params.local_condition_label_dim,
                                                       use_global_condition=params.use_global_condition,
                                                       global_condition_cardinality=params.global_condition_cardinality)

            wavenet = ProbabilityParameterEstimator(
                params.filter_width, params.residual_channels, params.dilations,
                params.skip_channels, params.out_channels,
                params.use_causal_conv_bias, params.use_filter_gate_bias, params.use_output_bias,
                params.use_skip_bias, params.use_postprocessing1_bias, params.use_postprocessing2_bias)

            X = labels.waveform[:, :-1]  # input
            Y = labels.waveform[:, 1:]  # target
            local_condition = features.mel[:, 1:, :]

            global_condition = features.global_condition[:, 1:] if params.use_global_condition else None

            H = condition_projection(local_condition, global_condition=global_condition)

            global_step = tf.train.get_global_step()

            if is_training:
                probability_params, _ = wavenet((X, H), sequential_inference_mode=False)
                loss = discretized_mix_logistic_loss(Y, probability_params, params.quantization_levels,
                                                     params.n_logistic_mix)
                lr = get_learning_rate(params.learning_rate_method, global_step,
                                       params={"learning_rate": params.learning_rate,
                                               "decay_steps": params.decay_steps,
                                               "decay_rate": params.decay_rate})
                optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=params.adam_beta1,
                                                   beta2=params.adam_beta2, epsilon=params.adam_eps)

                gradients, variables = zip(*optimizer.compute_gradients(loss))
                clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
                train_op = optimizer.apply_gradients(zip(clipped_gradients, variables), global_step=global_step)
                return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

            if is_validation:
                probability_params, _ = wavenet((X, H), sequential_inference_mode=False)
                loss = discretized_mix_logistic_loss(Y, probability_params, params.quantization_levels,
                                                     params.n_logistic_mix)
                return tf.estimator.EstimatorSpec(mode, loss=loss)

            if is_prediction:
                # ToDo: implement prediction
                return tf.estimator.EstimatorSpec(mode, predictions={
                    "id": features.id,
                    "key": features.key,
                    "mel": features.mel,
                    "text": features.text,
                })

        super(WaveNetModel, self).__init__(
            model_fn=model_fn, model_dir=model_dir, config=config,
            params=params, warm_start_from=warm_start_from)
