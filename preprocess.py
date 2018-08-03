# coding: utf-8
"""
Preprocess dataset
usage: preprocess.py [options] <name> <in_dir> <out_dir>


options:
    -h, --help               Show help message.

"""

from docopt import docopt
from datasets.corpus import corpus_factory
from hparams import hparams
from pyspark import SparkContext

if __name__ == "__main__":
    args = docopt(__doc__)
    name = args["<name>"]
    in_dir = args["<in_dir>"]
    out_dir = args["<out_dir>"]

    assert name in ["ljspeech"]
    corpus_instance = corpus_factory(name, in_dir, out_dir, hparams)

    sc = SparkContext()
    rdd = corpus_instance.text_and_audio_path_rdd(sc)
    result = corpus_instance.process_data(rdd).collect()
    print(len(result))