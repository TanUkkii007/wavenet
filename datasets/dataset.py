import tensorflow as tf
from collections import namedtuple
from abc import abstractmethod
from utils.tfrecord import decode_preprocessed_data, parse_preprocessed_data, PreprocessedData


class SourceData(namedtuple("SourceData",
                            ["id", "key", "mel", "mel_length", "mel_width", "text"])):
    pass


class TargetData(
    namedtuple("TargetData",
               ["id", "key", "waveform", "waveform_length"])):
    pass


class SourceDataForPrediction(namedtuple("SourceDataForPrediction",
                                         ["id", "key", "mel", "mel_length", "mel_width", "text", "waveform",
                                          "waveform_length"])):
    pass


def upsample_condition(condition, condition_length, times):
    last_dim = tf.shape(condition)[1]
    condition = tf.expand_dims(condition, 1)
    condition = tf.tile(condition, multiples=tf.convert_to_tensor([1, 1, times]))
    condition = tf.reshape(condition, shape=[-1, last_dim])
    condition_length = condition_length * times
    return condition, condition_length


def pad_waveform(wav, wav_length, condition_length):
    pad_length = condition_length - wav_length % condition_length
    wav = tf.pad(wav, paddings=tf.convert_to_tensor([[0, pad_length], [0, 0]]))
    wav_length = wav_length + pad_length
    return wav, wav_length


class DatasetSource:

    def __init__(self, dataset, hparams):
        self._dataset = dataset
        self._hparams = hparams

    @staticmethod
    def create_from_tfrecord_files(dataset_files, hparams, cycle_length=4,
                                   buffer_output_elements=None,
                                   prefetch_input_elements=None):
        dataset = tf.data.Dataset.from_generator(lambda: dataset_files, tf.string, tf.TensorShape([]))
        dataset = dataset.apply(tf.contrib.data.parallel_interleave(
            lambda filename: tf.data.TFRecordDataset(filename),
            cycle_length, sloppy=False,
            buffer_output_elements=buffer_output_elements,
            prefetch_input_elements=prefetch_input_elements))
        return DatasetSource(dataset, hparams)

    @property
    def hparams(self):
        return self._hparams

    def make_source_and_target(self):
        def make_pair(d: PreprocessedData):
            # ToDo: pad at the preprocessing time is more accurate
            waveform, waveform_length = pad_waveform(tf.expand_dims(d.waveform, axis=1), d.waveform_length,
                                                     d.mel_length)
            times = waveform_length // d.mel_length
            mel, mel_length = upsample_condition(d.mel, d.mel_length, times)
            source = SourceData(id=d.id,
                                key=d.key,
                                mel=mel,
                                mel_length=mel_length,
                                mel_width=d.mel_width,
                                text=d.text)
            target = TargetData(id=d.id,
                                key=d.key,
                                waveform=waveform,
                                waveform_length=waveform_length)
            return source, target

        zipped = self._decode_data().map(make_pair)
        return ZippedDataset(zipped, self.hparams)

    def _decode_data(self):
        return self._dataset.map(lambda d: decode_preprocessed_data(parse_preprocessed_data(d)))


class DatasetBase:

    @abstractmethod
    def apply(self, dataset, hparams):
        raise NotImplementedError("apply")

    @property
    @abstractmethod
    def dataset(self):
        raise NotImplementedError("dataset")

    @property
    @abstractmethod
    def hparams(self):
        raise NotImplementedError("hparams")

    def filter(self, predicate):
        return self.apply(self.dataset.filter(predicate), self.hparams)

    def filter_by_max_output_length(self):
        def predicate(s, t: PreprocessedData):
            max_output_length = self.hparams.max_output_length
            return tf.less_equal(t.waveform_length, max_output_length)

        return self.filter(predicate)

    def shuffle(self, buffer_size):
        return self.apply(self.dataset.shuffle(buffer_size), self.hparams)

    def repeat(self, count=None):
        return self.apply(self.dataset.repeat(count), self.hparams)

    def shuffle_and_repeat(self, buffer_size, count=None):
        dataset = self.dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size, count))
        return self.apply(dataset, self.hparams)


class ZippedDataset(DatasetBase):

    def __init__(self, dataset, hparams):
        self._dataset = dataset
        self._hparams = hparams

    def apply(self, dataset, hparams):
        return ZippedDataset(dataset, hparams)

    @property
    def dataset(self):
        return self._dataset

    @property
    def hparams(self):
        return self._hparams

    def group_by_batch(self, batch_size=None):
        batch_size = batch_size if batch_size is not None else self.hparams.batch_size
        approx_min_target_length = self.hparams.approx_min_target_length
        bucket_width = self.hparams.batch_bucket_width
        num_buckets = self.hparams.batch_num_buckets

        def key_func(unused_source, target):
            target_length = tf.minimum(target.waveform_length - approx_min_target_length, 0)
            bucket_id = target_length // bucket_width
            return tf.minimum(tf.to_int64(num_buckets), bucket_id)

        def reduce_func(unused_key, window: tf.data.Dataset):
            return window.padded_batch(batch_size, padded_shapes=(
                SourceData(
                    id=tf.TensorShape([]),
                    key=tf.TensorShape([]),
                    mel=tf.TensorShape([None, self.hparams.num_mels]),
                    mel_length=tf.TensorShape([]),
                    mel_width=tf.TensorShape([]),
                    text=tf.TensorShape([]),
                ),
                TargetData(
                    id=tf.TensorShape([]),
                    key=tf.TensorShape([]),
                    waveform=tf.TensorShape([None, 1]),
                    waveform_length=tf.TensorShape([]),
                )), padding_values=(
                SourceData(
                    id=tf.to_int64(0),
                    key="",
                    mel=tf.to_float(0),
                    mel_length=tf.to_int64(0),
                    mel_width=tf.to_int64(0),
                    text="",
                ),
                TargetData(
                    id=tf.to_int64(0),
                    key="",
                    waveform=tf.to_float(0),
                    waveform_length=tf.to_int64(0),
                )))

        batched = self.dataset.apply(tf.contrib.data.group_by_window(key_func,
                                                                     reduce_func,
                                                                     window_size=batch_size * 5))
        return BatchedDataset(batched, self.hparams)


class BatchedDataset(DatasetBase):

    def __init__(self, dataset: tf.data.Dataset, hparams):
        self._dataset = dataset
        self._hparams = hparams

    def apply(self, dataset, hparams):
        return BatchedDataset(dataset, hparams)

    @property
    def dataset(self):
        return self._dataset

    @property
    def hparams(self):
        return self._hparams

    def prefetch(self, buffer_size):
        return self.apply(self.dataset.prefetch(buffer_size), self.hparams)

    def arrange_for_prediction(self):
        def convert(s: SourceData, t: TargetData):
            return SourceDataForPrediction(
                id=s.id,
                key=s.key,
                mel=s.mel,
                mel_length=s.mel_length,
                mel_width=s.mel_width,
                text=s.text,
                waveform=t.waveform,
                waveform_length=t.waveform_length,
            ), t

        return self.apply(self.dataset.map(convert), self.hparams)
