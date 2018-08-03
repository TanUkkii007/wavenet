import librosa
import numpy as np
import tensorflow as tf
import scipy


class Audio:
    def __init__(self, hparams):
        self.hparams = hparams
        self._mel_basis = self._build_mel_basis()

    def _build_mel_basis(self):
        n_fft = (self.hparams.num_freq - 1) * 2
        return librosa.filters.mel(self.hparams.sample_rate, n_fft, n_mels=self.hparams.num_mels)

    def load_wav(self, path):
        return librosa.core.load(path, sr=self.hparams.sample_rate)[0]

    def save_wav(self, wav, path):
        wav = wav * 32767 / max(0.01, np.max(np.abs(wav)))
        scipy.io.wavfile.write(path, self.hparams.sample_rate, wav.astype(np.int16))

    def spectrogram(self, y):
        D = self._stft(y)
        S = self._amp_to_db(np.abs(D)) - self.hparams.ref_level_db
        return self._normalize(S)

    def melspectrogram(self, y):
        D = self._stft(y)
        S = self._amp_to_db(self._linear_to_mel(np.abs(D))) - self.hparams.ref_level_db
        return self._normalize(S)

    def _stft(self, y):
        n_fft, hop_length, win_length = self._stft_parameters()
        return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

    def _stft_tf(self, signals):
        n_fft, hop_length, win_length = self._stft_parameters()
        return tf.contrib.signal.stft(signals, win_length, hop_length, n_fft)

    def _istft_tf(self, stfts):
        n_fft, hop_length, win_length = self._stft_parameters()
        return tf.contrib.signal.inverse_stft(stfts, win_length, hop_length, n_fft)

    def _stft_parameters(self):
        n_fft = (self.hparams.num_freq - 1) * 2
        hop_length = int(self.hparams.frame_shift_ms / 1000 * self.hparams.sample_rate)
        win_length = int(self.hparams.frame_length_ms / 1000 * self.hparams.sample_rate)
        return n_fft, hop_length, win_length

    def _linear_to_mel(self, spectrogram):
        return np.dot(self._mel_basis, spectrogram)

    def _amp_to_db(self, x):
        return 20 * np.log10(np.maximum(1e-5, x))

    def _db_to_amp_tf(self, x):
        return tf.pow(tf.ones(tf.shape(x)) * 10.0, x * 0.05)

    def _normalize(self, S):
        return (S - self.hparams.min_level_db) / -self.hparams.min_level_db
