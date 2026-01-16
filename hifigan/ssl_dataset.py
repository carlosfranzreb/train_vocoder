import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor
import torch.utils.data
from torchcodec.decoders import AudioDecoder

from .mel_utils import LogMelSpectrogram


def load_wav(path: str, sr: int) -> Tensor:
    x = AudioDecoder(path, sample_rate=sr).get_all_samples().data
    return x


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def get_dataset_filelist(a):
    train_df = pd.read_csv(a.input_training_file)
    valid_df = pd.read_csv(a.input_validation_file)
    return train_df, valid_df


class SslDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        training_files,
        segment_size,
        n_fft,
        num_mels,
        hop_size,
        win_size,
        sampling_rate,
        fmin,
        fmax,
        split=True,
        shuffle=True,
        n_cache_reuse=1,
        device=None,
        fmax_loss=None,
        fine_tuning=False,
        audio_root_path=None,
        feat_root_path=None,
    ):
        self.audio_files = training_files
        if shuffle:
            self.audio_files = self.audio_files.sample(frac=1, random_state=1234)

        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.fine_tuning = fine_tuning
        self.audio_root_path = Path(audio_root_path)
        self.feat_root_path = Path(feat_root_path)
        self.melspec = LogMelSpectrogram(
            n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax
        )

    def __getitem__(self, index):
        row = self.audio_files.iloc[index]
        if self._cache_ref_count == 0:
            audio = load_wav(self.audio_root_path / row.audio_path, self.sampling_rate)
            self.cached_wav = audio
            self._cache_ref_count = self.n_cache_reuse
        else:
            audio = self.cached_wav
            self._cache_ref_count -= 1

        mel = torch.load(
            self.feat_root_path / row.feat_path, map_location="cpu"
        ).float()

        if len(mel.shape) < 3:
            mel = mel.unsqueeze(0)  # (1, seq_len, dim)

        if self.split:
            frames_per_seg = math.ceil(self.segment_size / self.hop_size)

            if audio.size(1) >= self.segment_size:
                mel_start = random.randint(0, mel.size(1) - frames_per_seg - 1)
                mel = mel[:, mel_start : mel_start + frames_per_seg, :]
                audio = audio[
                    :,
                    mel_start
                    * self.hop_size : (mel_start + frames_per_seg)
                    * self.hop_size,
                ]
            else:
                mel = torch.nn.functional.pad(
                    mel, (0, 0, 0, frames_per_seg - mel.size(2)), "constant"
                )
                audio = torch.nn.functional.pad(
                    audio, (0, self.segment_size - audio.size(1)), "constant"
                )

        # ground-truth mel-spectrogram
        mel_gt = self.melspec(audio).squeeze()

        return (mel.squeeze(), audio.squeeze(0), str(row.audio_path), mel_gt)

    def __len__(self):
        return len(self.audio_files)
