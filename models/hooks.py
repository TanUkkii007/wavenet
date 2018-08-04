import tensorflow as tf
import os
from utils.metrics import plot_wav


class MetricsSaver(tf.train.SessionRunHook):

    def __init__(self, global_step_tensor, predicted_audio_tensor, ground_truth_audio_tensor,
                 key_tensor, text_tensor, save_steps,
                 mode, hparams, writer: tf.summary.FileWriter):
        self.global_step_tensor = global_step_tensor
        self.predicted_audio_tensor = predicted_audio_tensor
        self.ground_truth_audio_tensor = ground_truth_audio_tensor
        self.key_tensor = key_tensor
        self.text_tensor = text_tensor
        self.save_steps = save_steps
        self.mode = mode
        self.hparams = hparams
        self.writer = writer

    def before_run(self, run_context):
        return tf.train.SessionRunArgs({
            "global_step": self.global_step_tensor
        })

    def after_run(self,
                  run_context,
                  run_values):
        stale_global_step = run_values.results["global_step"]
        if (stale_global_step + 1) % self.save_steps == 0 or stale_global_step == 0:
            global_step_value, predicted_audios, ground_truth_audios, keys, texts = run_context.session.run(
                (self.global_step_tensor, self.predicted_audio_tensor,
                 self.ground_truth_audio_tensor, self.key_tensor, self.text_tensor))
            for key, text, predicted_audio, ground_truth_audio in zip(keys, texts, predicted_audios,
                                                                      ground_truth_audios):
                output_filename = "{}_result_step{:09d}_{}.png".format(self.mode,
                                                                         global_step_value, key.decode('utf-8'))
                output_path = os.path.join(self.writer.get_logdir(), "audio_" + output_filename)
                tf.logging.info("Saving a %s result for %d at %s", self.mode, global_step_value, output_path)
                plot_wav(output_path, predicted_audio, ground_truth_audio, key, global_step_value, text.decode('utf-8'),
                         self.hparams.sample_rate)
