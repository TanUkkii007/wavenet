import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from librosa.display import waveplot


def plot_wav(path, y_hat, y_target, key, global_step, text, sample_rate):
    plt.figure(figsize=(16, 6))
    plt.subplot(2, 1, 1)
    waveplot(y_target, sr=sample_rate)
    plt.subplot(2, 1, 2)
    waveplot(y_hat, sr=sample_rate)
    plt.tight_layout()
    plt.suptitle(f"record: {key}, global step: {global_step}, input text: {str(text)}")
    plt.savefig(path, format="png")
    plt.close()