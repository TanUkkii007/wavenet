import tensorflow as tf


def get_learning_rate(name, global_step, params):
    name = name.lower()

    if name == 'constant':
        return params['learning_rate']
    elif name == 'exponential':
        # decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
        return tf.train.exponential_decay(global_step=global_step, **params)
    elif name == 'natural_exp':
        # decayed_learning_rate = learning_rate * exp(-decay_rate * global_step)
        return tf.train.natural_exp_decay(global_step=global_step, **params)
    elif name == 'inverse_time':
        # decayed_learning_rate = learning_rate / (1 + decay_rate * t)
        return tf.train.inverse_time_decay(global_step=global_step, **params)
    elif name == 'piecewise_constant':
        return tf.train.piecewise_constant(global_step=global_step, **params)
    elif name == 'polynomial':
        # decayed_learning_rate = (learning_rate - end_learning_rate) *
        #            (1 - global_step / decay_steps) ^ (power) + end_learning_rate
        return tf.train.polynomial_decay(global_step=global_step, **params)
    else:
        raise Exception('The learning rate method ' + name + ' has not been implemented.')
