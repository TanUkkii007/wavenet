"""Trainining script for seq2seq text-to-speech synthesis model.
Usage: train.py [options]

Options:
    --data-root=<dir>              Directory contains preprocessed features.
    --checkpoint-dir=<dir>         Directory where to save model checkpoints.
    --hparams=<parmas>             Hyper parameters. [default: ].
    --dataset=<name>               Dataset name.
    --training-list-file=<path>    Dataset file list for training.
    --validation-list-file=<path>  Dataset file list for validation.
    --log-file=<path>              Log file path.
    -h, --help                     Show this help message and exit
"""

from docopt import docopt
import tensorflow as tf
from random import shuffle
import os
import logging
from multiprocessing import cpu_count
from datasets.dataset import DatasetSource
from models.wavenet import WaveNetModel
from hparams import hparams, hparams_debug_string


def train_and_evaluate(hparams, model_dir, train_files, eval_files):
    interleave_parallelism = get_parallelism(hparams.interleave_cycle_length_cpu_factor,
                                             hparams.interleave_cycle_length_min,
                                             hparams.interleave_cycle_length_max)

    tf.logging.info("Interleave parallelism is %d.", interleave_parallelism)

    def train_input_fn():
        train_files_copy = list(train_files)
        shuffle(train_files_copy)
        dataset = DatasetSource.create_from_tfrecord_files(train_files_copy, hparams,
                                                           cycle_length=interleave_parallelism,
                                                           buffer_output_elements=hparams.interleave_buffer_output_elements,
                                                           prefetch_input_elements=hparams.interleave_prefetch_input_elements)
        batched = dataset.make_source_and_target().filter_by_max_output_length().shuffle_and_repeat(
            hparams.suffle_buffer_size).group_by_batch().prefetch(hparams.prefetch_buffer_size)
        return batched.dataset

    def eval_input_fn():
        eval_files_copy = list(eval_files)
        shuffle(eval_files_copy)

        eval_files_copy = tf.data.TFRecordDataset(eval_files_copy)

        dataset = DatasetSource(eval_files_copy, hparams)
        dataset = dataset.make_source_and_target().filter_by_max_output_length().repeat().group_by_batch(batch_size=1)
        return dataset.dataset

    run_config = tf.estimator.RunConfig(save_summary_steps=hparams.save_summary_steps,
                                        log_step_count_steps=hparams.log_step_count_steps)
    estimator = WaveNetModel(hparams, model_dir, config=run_config)

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn,
                                      steps=hparams.num_evaluation_steps,
                                      throttle_secs=hparams.eval_throttle_secs,
                                      start_delay_secs=hparams.eval_start_delay_secs)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def get_parallelism(factor, min_value, max_value):
    return min(max(int(cpu_count() * factor), min_value), max_value)


def load_key_list(path):
    with open(path, mode="r", encoding="utf-8") as f:
        for l in f:
            yield l.rstrip("\n")


def main():
    args = docopt(__doc__)
    print("Command line args:\n", args)
    checkpoint_dir = args["--checkpoint-dir"]
    data_root = args["--data-root"]
    dataset_name = args["--dataset"]
    training_list_file = args["--training-list-file"]
    validation_list_file = args["--validation-list-file"]
    log_file = args["--log-file"]

    hparams.parse(args["--hparams"])

    if log_file is not None:
        log = logging.getLogger("tensorflow")
        log.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        log.addHandler(fh)

    tf.logging.set_verbosity(tf.logging.INFO)

    tf.logging.info(hparams_debug_string())

    training_list = list(load_key_list(training_list_file))
    validation_list = list(load_key_list(validation_list_file))

    training_files = [os.path.join(data_root, f"{key}.tfrecord") for key in
                      training_list]
    validation_files = [os.path.join(data_root, f"{key}.tfrecord") for key in
                        validation_list]

    tf.logging.set_verbosity(tf.logging.INFO)
    train_and_evaluate(hparams,
                       checkpoint_dir,
                       training_files,
                       validation_files)


if __name__ == '__main__':
    main()
