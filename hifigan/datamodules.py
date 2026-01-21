import math
import random
from io import BytesIO
from dataclasses import dataclass
import logging
import os

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchcodec.decoders import AudioDecoder
from omegaconf import DictConfig
import webdataset as wds


def decode_pt(pt: bytes) -> Tensor:
    return torch.load(BytesIO(pt), weights_only=False)


class WdsAudioDecoder:
    def __init__(self, sample_rate: int):
        self.sr = sample_rate

    def __call__(self, audio: bytes) -> Tensor:
        encoded_data = torch.frombuffer(audio, dtype=torch.uint8).clone()
        return AudioDecoder(encoded_data, sample_rate=self.sr).get_all_samples().data


def is_not_metadata(sample: str) -> bool:
    fname = sample["__key__"].split("/")[-1]
    if len(fname) == 0:
        return False

    first_underscore = fname[0] == "_"
    second_underscore = fname[1] == "_"

    is_metadata = first_underscore and not second_underscore
    return not is_metadata


def create_dataloader(
    tar_dir: str,
    config: DictConfig,
    logger: logging.Logger,
    shuffle: bool = True,
) -> DataLoader:
    tar_paths = [
        os.path.join(tar_dir, f)
        for f in os.listdir(tar_dir)
        if f.endswith(".tar")
    ]
    logger.info(
        f"Creating dataloader with {len(tar_paths)} tar files and shuffle={shuffle}"
    )
    dataset = (
        wds.WebDataset(tar_paths)
        .select(is_not_metadata)
        .decode(
            wds.handle_extension("pt", decode_pt),
            wds.handle_extension(config.audio_ext, WdsAudioDecoder(config.sample_rate)),
        )
    )
    if shuffle:
        dataset = dataset.shuffle(config.n_shuffle)

    n_workers = config.num_workers
    if len(tar_paths) < n_workers:
        logger.warning(
            f"There are more workers ({n_workers}) than tars ({len(tar_paths)})"
        )
        n_workers = len(tar_paths)

    return DataLoader(
        dataset,
        collate_fn=Collator(
            config.segment_size, config.hifigan.hop_size, config.audio_ext
        ),
        num_workers=n_workers,
        batch_size=config.batch_size,
        pin_memory=config.device == "cuda",
        persistent_workers=config.num_workers > 0,
        drop_last=True,
    )


@dataclass
class DataBatch:
    ssl: Tensor
    audio: Tensor
    fnames: list[str]


class Collator:
    """
    Takes items from the webdataset (pairs of SSL features and their corresponding
    audios), segments them given a segment size and returns them.
    """

    def __init__(self, segment_size: int, hop_size: int, audio_ext: str):
        self.segment_size = segment_size
        self.audio_ext = audio_ext
        self.fps = math.ceil(segment_size / hop_size)
        self.hop_size = hop_size

    def __call__(self, batch: list[dict]) -> DataBatch:
        all_feats, all_audios, all_fnames = list(), list(), list()
        for item in batch:
            feats, audio = self.segment(item["pt"], item[self.audio_ext])
            all_feats.append(feats)
            all_audios.append(audio)
            all_fnames.append(item["__key__"])

        all_feats = pad_sequence(all_feats, batch_first=True)
        all_audios = pad_sequence(all_audios, batch_first=True)

        return DataBatch(all_feats, all_audios, all_fnames)

    def segment(self, feats: Tensor, audio: Tensor) -> tuple[Tensor, Tensor]:
        """
        Extracts a segment from both data sources, with a random start.

        Args:
            - feats: shape (n_vecs, vec_dim)
            - audio: shape (1, n_samples)

        Returns:
            - feats: shape (self.fps, vec_dim)
            - audio: shape(1, self.fps * self.hop_size = ~self.segment_size)
        """
        if audio.shape[1] >= self.segment_size:
            start = random.randint(0, feats.shape[0] - self.fps - 1)
            feats = feats[start : start + self.fps, :]
            audio = audio[
                :,
                start * self.hop_size : (start + self.fps) * self.hop_size,
            ]
        else:
            feats = torch.nn.functional.pad(
                feats, (0, 0, 0, self.fps - feats.shape[0]), "constant"
            )
            audio = torch.nn.functional.pad(
                audio, (0, (self.fps * self.hop_size) - audio.shape[1]), "constant"
            )

        return feats, audio
