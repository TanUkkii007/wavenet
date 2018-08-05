"""synthesize waveform
Usage: predict.py [options]

Options:
    --data-root=<dir>               Directory contains preprocessed features.
    --checkpoint-dir=<dir>          Directory where to save model checkpoints [default: checkpoints].
    --hparams=<parmas>              Hyper parameters. [default: ].
    --dataset=<name>                Dataset name.
    --test-list-file=<path>         Dataset file list for test.
    --checkpoint=<path>             Restore model from checkpoint path if given.
    --output-dir=<path>             Output directory.
    -h, --help                      Show this help message and exit
"""

from docopt import docopt
import tensorflow as tf
import os
from collections import namedtuple
from hparams import hparams, hparams_debug_string
from utils.audio import Audio
from utils.metrics import plot_wav
from datasets.dataset import DatasetSource
from models.wavenet import WaveNetModel


class PredictedAudio(
    namedtuple("PredictedAudio",
               ["id", "key", "predicted_waveform", "ground_truth_waveform", "mel", "text"])):
    pass


def predict(hparams,
            model_dir, checkpoint_path, output_dir, test_files):
    audio = Audio(hparams)

    def predict_input_fn():
        records = tf.data.TFRecordDataset(list(test_files))
        dataset = DatasetSource(records, hparams)
        batched = dataset.make_source_and_target().group_by_batch(
            batch_size=1).arrange_for_prediction()
        return batched.dataset

    estimator = WaveNetModel(hparams, model_dir)

    predictions = map(
        lambda p: PredictedAudio(p["id"], p["key"], p["predicted_waveform"], p["ground_truth_waveform"], p["mel"],
                                 p["text"]),
        estimator.predict(predict_input_fn, checkpoint_path=checkpoint_path))

    for v in predictions:
        key = v.key.decode('utf-8')
        audio_filename = f"{key}.wav"
        audio_filepath = os.path.join(output_dir, audio_filename)
        tf.logging.info(f"Saving {audio_filepath}")
        audio.save_wav(v.predicted_waveform, audio_filepath)
        png_filename = f"{key}.png"
        png_filepath = os.path.join(output_dir, png_filename)
        tf.logging.info(f"Saving {png_filepath}")
        # ToDo: pass global step
        plot_wav(png_filepath, v.predicted_waveform, v.ground_truth_waveform, key, 0, v.text.decode('utf-8'),
                 hparams.sample_rate)


def load_key_list(path):
    with open(path, mode="r", encoding="utf-8") as f:
        for l in f:
            yield l.rstrip("\n")


def main():
    args = docopt(__doc__)
    print("Command line args:\n", args)
    checkpoint_dir = args["--checkpoint-dir"]
    checkpoint_path = args["--checkpoint"]
    data_root = args["--data-root"]
    output_dir = args["--output-dir"]
    dataset_name = args["--dataset"]
    test_list_filepath = args["--test-list-file"]

    tf.logging.set_verbosity(tf.logging.INFO)

    hparams.parse(args["--hparams"])
    tf.logging.info(hparams_debug_string())

    test_list = list(load_key_list(test_list_filepath))

    test_files = [os.path.join(data_root, f"{key}.tfrecord") for key in
                  test_list]

    predict(hparams,
            checkpoint_dir,
            checkpoint_path,
            output_dir,
            test_files)


if __name__ == '__main__':
    main()
