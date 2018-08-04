import tensorflow as tf


def _log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    axis = len(x.get_shape()) - 1
    m = tf.reduce_max(x, axis)
    m2 = tf.reduce_max(x, axis, keepdims=True)
    return m + tf.log(tf.reduce_sum(tf.exp(x - m2), axis))


def _log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow """
    axis = len(x.get_shape()) - 1
    m = tf.reduce_max(x, axis=axis, keepdims=True)
    return m + tf.log(tf.reduce_sum(tf.exp(x - m), axis=axis, keepdims=True))


def discretized_mix_logistic_loss(X, probability_params, quantization_levels, nr_mix, sum_all=True):
    """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
    q = quantization_levels - 1.0
    half_q = q / 2.0

    X_shape = tf.shape(X)
    # here and below: unpacking the params of the mixture of logistics
    logit_probs = probability_params[:, :, :nr_mix]
    means = probability_params[:, :, nr_mix:2 * nr_mix]
    log_scales = tf.maximum(probability_params[:, :, 2 * nr_mix:3 * nr_mix], -14.0)

    X = X + tf.zeros((X_shape[0], X_shape[1], nr_mix))
    centered_X = X - means
    inv_stdv = tf.exp(-log_scales)
    plus_in = inv_stdv * (centered_X + 1. / q)
    cdf_plus = tf.nn.sigmoid(plus_in)
    min_in = inv_stdv * (centered_X - 1. / q)
    cdf_min = tf.nn.sigmoid(min_in)
    # log probability for edge case of 0 (before scaling)
    log_cdf_plus = plus_in - tf.nn.softplus(plus_in)
    # log probability for edge case of q (before scaling)
    log_one_minus_cdf_min = -tf.nn.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases

    # robust version, that still works if probabilities are below 1e-6 (which never happens in our code)
    # tensorflow backpropagates through tf.select() by multiplying with zero instead of selecting: this requires use to use some ugly tricks to avoid potential NaNs
    # the 1e-12 in tf.maximum(cdf_delta, 1e-12) is never actually used as output, it's purely there to get around the tf.select() gradient issue
    # if the probability on a sub-pixel is below 1e-5, we use an approximation
    # based on the assumption that the log-density is constant in the bin of
    # the observed sub-pixel value
    max_threshold = (q - 1.0) / q
    min_threshold = -max_threshold

    log_probs = tf.where(X < min_threshold, log_cdf_plus,
                         tf.where(X > max_threshold, log_one_minus_cdf_min, tf.log(tf.maximum(cdf_delta, 1e-14))))
    log_probs = log_probs + _log_prob_from_logits(logit_probs)
    if sum_all:
        return -tf.reduce_mean(_log_sum_exp(log_probs))
    else:
        return -log_probs


def sample_from_discretized_mix_logistic(probability_params, nr_mix):
    p_shape = tf.shape(probability_params)
    # unpack parameters
    logit_probs = probability_params[:, :, :nr_mix]

    # sample mixture indicator from softmax
    index = tf.argmax(
        logit_probs - tf.log(-tf.log(tf.random_uniform(shape=tf.shape(logit_probs), minval=1e-6, maxval=1.0 - 1e-6))),
        axis=2)
    index = tf.one_hot(index, depth=nr_mix, dtype=probability_params.dtype)

    # select logistic parameters
    means = probability_params[:, :, nr_mix:2 * nr_mix]
    means = tf.reduce_sum(means * index, axis=-1)  # one hot selection and zero removal by reduce sum
    log_scales = probability_params[:, :, 2 * nr_mix:3 * nr_mix]
    log_scales = tf.reduce_sum(log_scales * index, axis=-1)  # one hot selection and zero removal by reduce sum
    log_scales = tf.maximum(log_scales, -14.0)

    # sample from logistic & clip to interval
    # we don't actually round to the nearest 16bit value when sampling
    u = tf.random_uniform(shape=tf.shape(means), minval=1e-5, maxval=1.0 - 1e-5)
    x = means + tf.exp(log_scales) * (tf.log(u) - tf.log(1.0 - u))

    return x
