import tensorflow as tf

hparams = tf.contrib.training.HParams(

    # Audio
    num_mels=80,
    num_freq=1025,
    sample_rate=16000,
    frame_length_ms=25.0,
    frame_shift_ms=5.0,
    min_level_db=-100,
    ref_level_db=20,

    # Model
    quantization_levels=256,
    n_logistic_mix=15,
    local_condition_label_dim=80,
    use_global_condition=False,
    global_condition_cardinality=0,
    filter_width=3,
    residual_channels=64,
    dilations=[1,2,4,8,16,32,64,128,256,512,
               1,2,4,8,16,32,64,128,256,512,
               1,2,4,8,16,32,64,128,256,512],
    skip_channels=128,
    out_channels=3*15,  # 3*n_logistic_mix
    use_causal_conv_bias=True,
    use_filter_gate_bias=True,
    use_output_bias=True,
    use_skip_bias=True,
    use_postprocessing1_bias=True,
    use_postprocessing2_bias=True,

    max_output_length=128000,

    # Training:
    batch_size=2,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_eps=1e-8,
    learning_rate_method="exponential",
    learning_rate=0.001,
    decay_steps=20000,
    decay_rate=0.99,
    save_summary_steps=50,
    log_step_count_steps=1,
    alignment_save_steps=100,
    approx_min_target_length=32000,
    suffle_buffer_size=64,
    batch_bucket_width=50,
    batch_num_buckets=50,
    interleave_cycle_length_cpu_factor=1.0,
    interleave_cycle_length_min=4,
    interleave_cycle_length_max=16,
    interleave_buffer_output_elements=200,
    interleave_prefetch_input_elements=200,
    prefetch_buffer_size=4,

    # Eval:
    num_evaluation_steps=32,
    eval_start_delay_secs=1800,
    eval_throttle_secs=4000,
)


def hparams_debug_string():
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
    return 'Hyperparameters:\n' + '\n'.join(hp)