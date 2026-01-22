# Train vocoders

This repository was forked from the kNN-VC repo, which in turn was forked from the HiFiGAN repo.
I have updated the codebase to conform to the new package versions like PyTorch. I have also introduced
webdataset, which reduces training time by 25%.

Currently, this repo follows exactly the procedure outlined by kNN-VC (see below).
With time, I would like to expand it to other vocoder training procedures.

## Inference

Check out the kNN-VC repo for how to perform inference.

## Training

We follow the typical encoder-converter-vocoder setup for voice conversion. The encoder is WavLM, the converter is k-nearest neighbors regression, and vocoder is HiFiGAN. The only component that requires training is the vocoder:

1. **WavLM encoder**: we simply use the pretrained WavLM-Large model and do not train it for any part of our work. We suggest checking out the original [WavLM repo](https://github.com/microsoft/unilm) to train your own SSL encoder.
2. **kNN conversion model**: kNN is non-parametric and does not require any training :)
3. **HiFiGAN vocoder**: we adapt the original [HiFiGAN author's repo](https://github.com/jik876/hifi-gan) for vocoding WavLM features. This is the only part which requires training.

### HiFiGAN training

1. **Precompute WavLM features of the vocoder dataset**: we provide a utility for this for the LibriSpeech dataset in `prematch_dataset.py`:

    ```bash
    usage: prematch_dataset.py [-h] --librispeech_path LIBRISPEECH_PATH
                            [--seed SEED] --out_path OUT_PATH [--device DEVICE]
                            [--topk TOPK] [--layer LAYER = 6]
                            [--prematch]
                            [--resume]
    ```

    where you can specify `--prematch` or not to determine whether to use prematching when generating features or not. For example, to generate the dataset used to train the prematched HiFiGAN from the paper:
    `python prematch_dataset.py --librispeech_path /path/to/librispeech/root --out_path /path/where/you/want/outputs/to/go --topk 4 --matching_layer 6 --synthesis_layer 6 --prematch`

2. **Tar dataset**: to improve training efficiency, we tar the dataset and use webdataset.

```python
python scripts/create_webdataset.py
    output_dir=ls-dev-clean_prematch
    audio_dir=/cfs/collections/librispeech/LibriSpeech/dev-clean/
    ssl_dir=librispeech_prematch/dev-clean/
    n_tars=20
```

3. **Train HiFiGAN**: we adapt the training script from the [original HiFiGAN repo](https://github.com/jik876/hifi-gan) to work for WavLM features in `hifigan/train.py`. To train a hifigan model on the features you produced above:

    ```bash
    python -m hifigan.train hifigan/config.yaml
    ```

    That's it! Once it is run up till 2.5M updates (or it starts to sound worse) you can stop training and use the trained checkpoint.

## Optimize train

Training on 1 H200:

- commit a51930398: 460s per epoch on /cfs
- commit a51930398: 160s per epoch on /tmp (increasing workers from 10 to 20, which I think has a bigger impact).

Training on 1 RTX 6000 (Ada):

- commit 07ccfa0: 120s per epoch with webdataset (same bs=32 and n_workers=20 as before)
- commit 07ccfa0: 120s per epoch with webdataset (n_workers=50)