import tensorflow as tf
import numpy as np
from collections import namedtuple
from collections.abc import Iterable


class PreprocessedData(namedtuple("PreprocessedData",
                                  ["id", "key", "waveform", "waveform_length", "mel", "mel_length", "mel_width",
                                   "text"])):
    pass


def bytes_feature(value):
    assert isinstance(value, Iterable)
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def int64_feature(value):
    assert isinstance(value, Iterable)
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def write_tfrecord(example: tf.train.Example, filename: str):
    with tf.python_io.TFRecordWriter(filename) as writer:
        writer.write(example.SerializeToString())


def write_preprocessed_data(id: int, key: str, waveform: np.ndarray, mel: np.ndarray, text: str, filename: str):
    raw_waveform = waveform.tostring()
    raw_mel = mel.tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'id': int64_feature([id]),
        'key': bytes_feature([key.encode('utf-8')]),
        'waveform': bytes_feature([raw_waveform]),
        'waveform_length': int64_feature([len(waveform)]),
        'mel': bytes_feature([raw_mel]),
        'mel_length': int64_feature([len(mel)]),
        'mel_width': int64_feature([mel.shape[1]]),
        'text': bytes_feature([text.encode('utf-8')]),
    }))
    write_tfrecord(example, filename)


def parse_preprocessed_data(proto):
    features = {
        'id': tf.FixedLenFeature((), tf.int64),
        'key': tf.FixedLenFeature((), tf.string),
        'waveform': tf.FixedLenFeature((), tf.string),
        'waveform_length': tf.FixedLenFeature((), tf.int64),
        'mel': tf.FixedLenFeature((), tf.string),
        'mel_length': tf.FixedLenFeature((), tf.int64),
        'mel_width': tf.FixedLenFeature((), tf.int64),
        'text': tf.FixedLenFeature((), tf.string),
    }
    parsed_features = tf.parse_single_example(proto, features)
    return parsed_features


def decode_preprocessed_data(parsed):
    waveform_length = parsed['waveform_length']
    waveform = tf.decode_raw(parsed['waveform'], tf.float32)
    mel_width = parsed['mel_width']
    mel_length = parsed['mel_length']
    mel = tf.decode_raw(parsed['mel'], tf.float32)
    return PreprocessedData(
        id=parsed['id'],
        key=parsed['key'],
        waveform=waveform,
        waveform_length=waveform_length,
        mel=tf.reshape(mel, shape=tf.stack([mel_length, mel_width], axis=0)),
        mel_width=mel_width,
        mel_length=mel_length,
        text=parsed['text'],
    )
