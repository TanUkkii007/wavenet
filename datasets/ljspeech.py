from pyspark import SparkContext, RDD
import numpy as np
import os
from collections import namedtuple
from utils import tfrecord
from utils.audio import Audio


class TextAndAudioPath(namedtuple("TextAndAudioPath", ["id", "key", "wav_path", "text"])):
    pass


class SourceAndTargetMetaData(namedtuple("TargetMetaData", ["id", "key", "n_samples", "n_frames", "filepath"])):
    pass


class MelMetaData(namedtuple("MelMetaData", ["n_frames", "sum_mel_powers"])):
    pass


class LJSpeech:

    def __init__(self, in_dir, out_dir, hparams):
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.audio = Audio(hparams)

    def text_and_audio_path_rdd(self, sc: SparkContext):
        return sc.parallelize(
            self._extract_all_text_and_audio_path())

    def process_data(self, rdd: RDD):
        return rdd.mapValues(self._process_source_and_target)

    def _extract_text_and_path(self, line, index):
        parts = line.strip().split('|')
        key = parts[0]
        wav_path = os.path.join(self.in_dir, 'wavs', '%s.wav' % parts[0])
        text = parts[2]
        return TextAndAudioPath(index, key, wav_path, text)

    def _extract_all_text_and_audio_path(self):
        index = 1
        with open(os.path.join(self.in_dir, 'metadata.csv'), mode='r', encoding='utf-8') as f:
            for line in f:
                extracted = self._extract_text_and_path(line, index)
                if extracted is not None:
                    yield (index, extracted)
                    index += 1

    def _process_source_and_target(self, paths: TextAndAudioPath):
        wav = self.audio.load_wav(paths.wav_path)
        n_samples = len(wav)
        mel_spectrogram = self.audio.melspectrogram(wav).astype(np.float32).T
        n_frames = mel_spectrogram.shape[0]
        filename = f"{paths.key}.tfrecord"
        filepath = os.path.join(self.out_dir, filename)
        tfrecord.write_preprocessed_data(paths.id, paths.key, wav, mel_spectrogram, paths.text, filepath)
        return SourceAndTargetMetaData(paths.id, paths.key, n_samples, n_frames, filepath)

    def _process_mel(self, paths: TextAndAudioPath):
        wav = self.audio.load_wav(paths.wav_path)
        mel_spectrogram = self.audio.melspectrogram(wav).astype(np.float32).T
        sum_mel_powers = np.sum(mel_spectrogram, axis=1)
        n_frames = mel_spectrogram.shape[0]
        return MelMetaData(n_frames, sum_mel_powers)


