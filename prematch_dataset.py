"""
If torchcodec, cannot find ffmpeg: export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchcodec.decoders import AudioDecoder
from tqdm import tqdm

from wavlm import init_wavlm_large

DOWNSAMPLE_FACTOR = 320
MIN_VECTORS_FOR_PREMATCH = 9000  # 3 minutes
LOGGER = logging.getLogger("prematch_dataset.log")

global feature_cache
feature_cache = {}


def make_df(root_path: Path, ext: str = ".flac") -> pd.DataFrame:

    LOGGER.info(f"Loading files from {root_path}")
    files = list((root_path).rglob("**/*" + ext))
    speakers = [f.stem.split("-")[0] for f in files]
    df = pd.DataFrame({"path": files, "speaker": speakers})
    LOGGER.info(f"Loaded {len(df)} files")

    return df


def main(args):
    df = make_df(Path(args.path), ext=args.ext)

    LOGGER.info(f"Loading wavlm.")
    wavlm = init_wavlm_large(pretrained=True, progress=True, device=args.device)
    wavlm.extract_from_layer = args.layer

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    extract(
        df,
        wavlm,
        args.device,
        args.prematch,
        args.topk,
        Path(args.path),
        Path(args.out_path),
        args.resume,
    )
    LOGGER.info("All done!")


@torch.inference_mode()
def get_features(path: Path, wavlm: nn.Module, device: str) -> Tensor:
    """
    Extracts WavLM features from the given audio path, and returns them as a single
    tensor of shape (n_feats, feat_dim).
    """

    # load audio and ensure it's divisible by WavLM's featex
    x = AudioDecoder(path, sample_rate=16_000).get_all_samples().data
    n_pad = DOWNSAMPLE_FACTOR - (x.shape[-1] % DOWNSAMPLE_FACTOR)
    x = F.pad(x, (0, n_pad), value=0)

    # extract the representation of each layer
    x = x.to(device)
    features = wavlm.extract_features(
        x, output_layer=wavlm.extract_from_layer, ret_layer_results=True
    )[0][1][-1][0].squeeze(1)

    return features


def fast_cosine_dist(source_feats: Tensor, pool: Tensor) -> Tensor:
    """
    Receives two tensors of shape (n_feats_a, feat_dim), (n_feats_b, feat_dim) and
    returns the distances between all pairs of their features (n_feats_a, n_feats_b).
    """
    source_norms = torch.norm(source_feats, p=2, dim=-1)
    norms = torch.norm(pool, p=2, dim=-1)
    dotprod = (
        -torch.cdist(source_feats[None], pool[None], p=2)[0] ** 2
        + source_norms[:, None] ** 2
        + norms[None] ** 2
    )
    dotprod /= 2

    dists = 1 - (dotprod / (source_norms[:, None] * norms[None]))
    return dists


@torch.inference_mode()
def extract(
    df: pd.DataFrame,
    wavlm: nn.Module,
    device: str,
    prematch: bool,
    topk: int,
    ls_path: Path,
    out_path: Path,
    resume: bool,
):

    # iterate over all unique speakers
    for i, (speaker, group) in enumerate(
        tqdm(df.groupby("speaker"), total=df["speaker"].nunique())
    ):
        # extract features from all the speaker's utterances
        feats = list()
        dump_paths = list()
        for _, row in group.iterrows():

            rel_path = Path(row.path).relative_to(ls_path)
            target_path = (out_path / rel_path).with_suffix(".pt")
            if resume and target_path.is_file():
                LOGGER.warning(f"Features already exist for {rel_path}")
                target_path = None

            os.makedirs(target_path.parent, exist_ok=True)
            feats.append(get_features(row.path, wavlm, device))
            dump_paths.append(target_path)

        # do the pre-matching if needed, when there are enough target feats
        # TODO: create pool once, and then use masking to remove source_feats
        if prematch:
            matched_feats = list()
            for utt_idx in range(len(feats)):
                if resume and dump_paths[utt_idx] is None:
                    continue

                source_feats = feats[utt_idx]
                pool = torch.cat(
                    [feats[idx] for idx in range(len(feats)) if idx != utt_idx], dim=0
                )

                if pool.shape[0] < MIN_VECTORS_FOR_PREMATCH:
                    LOGGER.warning(
                        f"Not enough target vectors for {dump_paths[utt_idx]}"
                    )
                    matched_feats.append(source_feats)
                else:
                    dists = fast_cosine_dist(source_feats, pool)
                    best = dists.topk(k=topk, dim=-1, largest=False)
                    matched_feats.append(pool[best.indices].mean(dim=1))

            feats = matched_feats

        # dump the features and continue
        for dump_path, dump_feats in zip(dump_paths, feats):
            torch.save(dump_feats.cpu().half(), dump_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute matched wavlm features for a dataset"
    )

    parser.add_argument("path", type=str)
    parser.add_argument("out_path", type=str)
    parser.add_argument("--seed", default=123, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--ext", default=".flac", type=str)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--prematch", action="store_true", help="prematch")
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()
    main(args)
