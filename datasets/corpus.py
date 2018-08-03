from datasets.ljspeech import LJSpeech


def corpus_factory(name, in_dir, out_dir, hparams):
    if name == "ljspeech":
        return LJSpeech(in_dir, out_dir, hparams)
    else:
        raise ValueError(f"Unknown corpus: {name}")