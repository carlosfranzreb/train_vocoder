import argparse
import logging
import gc
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from fastprogress.fastprogress import progress_bar
from torch import Tensor

from hubconf import wavlm_large

DOWNSAMPLE_FACTOR = 320
MIN_VECTORS_FOR_PREMATCH = 9000  # 3 minutes
LOGGER = logging.getLogger("prematch_dataset.log")

global feature_cache
feature_cache = {}


def make_df(
    root_path: Path, folders: list[str] = None, ext: str = ".flac"
) -> pd.DataFrame:

    LOGGER.info(f"Loading files from {root_path}] with folders {folders}")
    if folders is None or len(folders) == 0:
        all_files = list((root_path).rglob("**/*" + ext))
    else:
        all_files = list()
        for f in folders:
            all_files.extend(list((root_path / f).rglob("**/*" + ext)))

    speakers = [f.stem.split("-")[0] for f in all_files]
    df = pd.DataFrame({"path": all_files, "speaker": speakers})
    LOGGER.info(f"Loaded {len(df)} files")

    return df


def main(args):
    device = torch.device(args.device)
    weights = (
        F.one_hot(torch.tensor(args.layer), num_classes=25).float().to(device)[:, None]
    )
    LOGGER.info(f"Matching weights: {weights.squeeze()}")
    df = make_df(Path(args.path), folders=args.folder, ext=args.ext)

    LOGGER.info(f"Loading wavlm.")
    wavlm = wavlm_large(pretrained=True, progress=True, device=args.device)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    extract(
        df,
        wavlm,
        args.device,
        Path(args.path),
        Path(args.out_path),
        weights,
        args.ext,
    )
    LOGGER.info("All done!")


def path2pools(
    path: Path,
    wavlm: nn.Module,
    weights: Tensor,
    device: str,
    ext: str,
) -> Tensor:
    """Given a waveform `path`, compute the matching pool"""

    uttrs_from_same_spk = sorted(list(path.parent.rglob("**/*" + ext)))
    uttrs_from_same_spk.remove(path)
    pool = list()
    for pth in uttrs_from_same_spk:
        if pth in feature_cache:
            feats = feature_cache[pth].float()  # (seq_len, dim)
        else:
            feats = get_full_features(pth, wavlm, device)
            feats = (feats * weights[:, None]).sum(dim=0)  # (seq_len, dim)
            feature_cache[pth] = feats.half().cpu()

        pool.append(feats.cpu())

    try:
        pool = torch.concat(pool, dim=0)
    except RuntimeError:
        LOGGER.warning(f"No matching pool available for file {path}")
        pool = torch.empty((1, 1024), device=device)

    return pool  # (N, dim)


@torch.inference_mode()
def get_full_features(path, wavlm, device):

    x, sr = torchaudio.load(path)
    if sr != 16000:
        x = torchaudio.transforms.Resample(sr, 16000)(x)

    # This does not work i.t.o the hifigan training.
    # x = F.pad(x, (DOWNSAMPLE_FACTOR//2, DOWNSAMPLE_FACTOR - DOWNSAMPLE_FACTOR//2), value=0)
    # This does.
    n_pad = DOWNSAMPLE_FACTOR - (x.shape[-1] % DOWNSAMPLE_FACTOR)
    x = F.pad(x, (0, n_pad), value=0)

    # extract the representation of each layer
    wav_input_16khz = x.to(device)
    rep, layer_results = wavlm.extract_features(
        wav_input_16khz, output_layer=wavlm.cfg.encoder_layers, ret_layer_results=True
    )[0]
    features = torch.cat(
        [x.transpose(0, 1) for x, _ in layer_results], dim=0
    )  # (n_layers, seq_len, dim)

    return features


def fast_cosine_dist(source_feats, pool):
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
    device,
    ls_path: Path,
    out_path: Path,
    weights: Tensor,
    ext: str,
):

    pb = progress_bar(df.iterrows(), total=len(df))

    for i, row in pb:
        rel_path = Path(row.path).relative_to(ls_path)
        targ_path = (out_path / rel_path).with_suffix(".pt")
        if args.resume:
            if targ_path.is_file():
                continue

        os.makedirs(targ_path.parent, exist_ok=True)

        if Path(row.path) in feature_cache:
            source_feats = feature_cache[Path(row.path)].float()
        else:
            source_feats = get_full_features(row.path, wavlm, device)
            source_feats = (source_feats * weights[:, None]).sum(
                dim=0
            )  # (seq_len, dim)

        pool = path2pools(row.path, wavlm, weights, device, ext)

        if not args.prematch or pool.shape[0] < MIN_VECTORS_FOR_PREMATCH:
            out_feats = source_feats.cpu()
        else:
            dists = fast_cosine_dist(source_feats.cpu(), pool.cpu()).cpu()
            best = dists.topk(k=args.topk, dim=-1, largest=False)  # (src_len, 4)
            out_feats = pool[best.indices].mean(dim=1)  # (N, dim)

        # save matched sequence
        if i < 3:
            LOGGER.info("Feature has shape: ", out_feats.shape)

        # 3. save
        torch.save(out_feats.cpu().half(), str(targ_path))
        if hasattr(pb, "child"):
            pb.child.comment = str(rel_path)
            pb.child.wait_for = min(pb.child.wait_for, 10)
            pb.main_bar.comment = str(rel_path)
        else:
            pb.wait_for = min(pb.wait_for, 10)

        pb.comment = str(rel_path)

        if i % 1000 == 0:
            LOGGER.info(f"Done {i:,d}/{len(df):,d}")
            feature_cache.clear()
            gc.collect()
            time.sleep(4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute matched wavlm features for a dataset"
    )

    parser.add_argument("--path", required=True, type=str)
    parser.add_argument("--folder", action="append")
    parser.add_argument("--seed", default=123, type=int)
    parser.add_argument("--out_path", required=True, type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--ext", default=".flac", type=str)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--prematch", action="store_true", help="prematch")
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()
    main(args)
