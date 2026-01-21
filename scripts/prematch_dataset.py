"""
! If torchcodec cannot find ffmpeg: export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib

This script extracts and pre-matches WavLM audio features from a dataset of utterances.

Pre-matching refers to a kNN regression across utterances from the same speaker to
create matched feature pairs, similar to those you would get from a real conversion
step on inference. Given a speaker's utterances, `prematch_feats`, for each utterance,

- finds the k nearest features from all *other* utterances of the same speaker
- Averages those k nearest neighbors to create a matched representation

For speakers with a lot of data, the data is split into chunks to avoid memory issues
while maintaining independent pre-matching within chunks. You can define how to chunk
the speaker's data with MAX_VECTORS_FOR_PREMATCH.

With MIN_VECTORS_FOR_PREMATCH, you can define a lower bound for the amount of data
required for pre-matching. Utterances for which the matching pool is lower than this
threshold are not pre-matched, and stored as-is.

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
MAX_VECTORS_FOR_PREMATCH = 24000  # 8 minutes

# create logger
LOGGER = logging.getLogger("prematch_dataset")
handler = logging.FileHandler("prematch_dataset.log")
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)


def make_df(root_path: Path, ext: str = ".flac") -> pd.DataFrame:

    LOGGER.info(f"Loading files from {root_path}")
    files = list((root_path).rglob("**/*" + ext))
    speakers = [f.stem.split("-")[0] for f in files]
    df = pd.DataFrame({"path": files, "speaker": speakers})
    LOGGER.info(f"Loaded {len(df)} files")

    return df


def main(args):
    LOGGER.info(f"Starting run with args {args}")
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
    tensor of shape (1, n_feats, feat_dim).
    """

    # load audio and ensure it's divisible by WavLM's featex
    x = AudioDecoder(path, sample_rate=16_000).get_all_samples().data
    n_pad = DOWNSAMPLE_FACTOR - (x.shape[-1] % DOWNSAMPLE_FACTOR)
    x = F.pad(x, (0, n_pad), value=0)

    # extract the representation of each layer
    x = x.to(device)
    features = (
        wavlm.extract_features(
            x, output_layer=wavlm.extract_from_layer, ret_layer_results=True
        )[0][1][-1][0]
        .squeeze(1)
        .unsqueeze(0)
    )

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
    for _, group in tqdm(df.groupby("speaker"), total=df["speaker"].nunique()):
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

        if prematch:
            # split data if there is enough for at least 2 disjoint parts
            feat_lens = torch.tensor([f.shape[0] for f in feats])
            sum_feats = torch.sum(feat_lens)
            if sum_feats > MAX_VECTORS_FOR_PREMATCH * 2:

                # find where to split
                n_splits = sum_feats // MAX_VECTORS_FOR_PREMATCH
                chunk_size = sum_feats // n_splits
                cumsum_feats = torch.cumsum(feat_lens, dim=0)

                # prematch each chunk
                feats_prematched = list()
                start_idx = 0
                for split_number in range(n_splits):
                    if split_number == n_splits - 1:
                        split_idx = len(feats)
                    else:
                        split_max = chunk_size * (split_number + 1)
                        split_idx = torch.argwhere(cumsum_feats > split_max)[0]

                    chunk = feats[start_idx:split_idx]
                    feats_prematched.extend(
                        prematch_feats(chunk, topk, dump_paths, resume)
                    )
                    start_idx = split_idx

                feats = feats_prematched

            else:
                feats = prematch_feats(feats, topk, dump_paths, resume)

        # dump the features and continue
        for dump_path, dump_feats in zip(dump_paths, feats):
            torch.save(dump_feats.cpu().half(), dump_path)


def prematch_feats(
    feats: list[Tensor], topk: int, dump_paths: list[str], resume: bool
) -> list[Tensor]:
    """
    Given the WavLM features of several utterances of the same speaker, pre-match
    them by doing a kNN regression where the utterance being regressed is removed
    from the pool.
    """

    # vectorize feats and keep track of the utterances
    device = feats[0].device
    n_utts = len(feats)

    # Pre-calculate offsets to avoid repeated masking
    offsets = torch.cumsum(torch.tensor([0] + [f.shape[0] for f in feats]), dim=0)

    utts = torch.hstack(
        [
            torch.ones(feats[idx].shape[0], dtype=torch.int) * idx
            for idx in range(len(feats))
        ]
    )
    feats = torch.vstack(feats)

    # compute cosine distances between all features
    dists = torch.zeros((feats.shape[0], feats.shape[0]), device=device)
    for idx_a in range(n_utts):
        start_a, end_a = offsets[idx_a], offsets[idx_a + 1]
        for idx_b in range(idx_a + 1, n_utts):
            start_b, end_b = offsets[idx_b], offsets[idx_b + 1]
            utt_dists = fast_cosine_dist(feats[start_a:end_a], feats[start_b:end_b])
            dists[start_a:end_a, start_b:end_b] = utt_dists
            dists[start_b:end_b, start_a:end_a] = utt_dists.T

    # do the pre-matching
    matched_feats = list()
    for utt_idx in torch.arange(n_utts):

        if resume and dump_paths[utt_idx] is None:
            continue

        utt_mask = utts == utt_idx
        target_feats = feats[~utt_mask]

        if target_feats.shape[0] < MIN_VECTORS_FOR_PREMATCH:
            LOGGER.warning(f"Not enough target vectors for {dump_paths[utt_idx]}")
            matched_feats.append(feats[utt_mask])
        else:
            utt_dists = dists[utt_mask][:, ~utt_mask]
            best = utt_dists.topk(k=topk, dim=-1, largest=False)
            matched_feats.append(target_feats[best.indices].mean(dim=1))

    return matched_feats


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
    parser.add_argument("--prematch", action="store_true")
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()
    main(args)
