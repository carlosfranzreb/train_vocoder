import argparse
import itertools
import os
import time
import logging
import subprocess
import math
import random
from io import BytesIO
from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
from torchcodec.decoders import AudioDecoder

from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import webdataset as wds

from .mel_utils import LogMelSpectrogram
from .models import (
    Generator,
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    discriminator_loss,
    feature_loss,
    generator_loss,
)
from .utils import AttrDict, load_checkpoint, save_checkpoint, scan_checkpoint

torch.backends.cudnn.benchmark = True


def decode_pt(pt: bytes) -> Tensor:
    return torch.load(BytesIO(pt), weights_only=False)


class WdsAudioDecoder:
    def __init__(self, sample_rate: int):
        self.sr = sample_rate

    def __call__(self, audio: bytes) -> Tensor:
        encoded_data = torch.frombuffer(audio, dtype=torch.uint8)
        return AudioDecoder(encoded_data, sample_rate=self.sr).get_all_samples().data


def is_not_metadata(sample: str) -> bool:
    fname = sample["__key__"].split("/")[-1]
    if len(fname) == 0:
        return False

    first_underscore = fname[0] == "_"
    second_underscore = fname[1] == "_"

    is_metadata = first_underscore and not second_underscore
    if is_metadata:
        print(sample)
    return not is_metadata


def create_dataloader(
    tar_file: str, config: DictConfig, shuffle: bool = True
) -> DataLoader:
    dataset = (
        wds.WebDataset(tar_file)
        .select(is_not_metadata)
        .decode(
            wds.handle_extension("pt", decode_pt),
            wds.handle_extension(config.audio_ext, WdsAudioDecoder(config.sample_rate)),
        )
    )
    if shuffle:
        dataset = dataset.shuffle(1_000)

    return DataLoader(
        dataset,
        collate_fn=Collator(
            config.segment_size, config.hifigan.hop_size, config.audio_ext
        ),
        num_workers=config.num_workers,
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


def train(config: AttrDict, logger: logging.Logger):

    # init models
    device = config.device
    generator = Generator(config.hifigan).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)

    # check if ckpt folder already exists and retrieve checkpoints
    os.makedirs(config.ckpt_path, exist_ok=True)
    logger.info("checkpoints directory : ", config.ckpt_path)
    if os.path.isdir(config.ckpt_path):
        cp_g = scan_checkpoint(config.ckpt_path, "g_")
        cp_do = scan_checkpoint(config.ckpt_path, "do_")

    # if ckpt folder is new, start training from scratch
    if cp_g is None or cp_do is None:
        steps = 0
        state_dict_do = None
        last_epoch = -1

    # otherwise, resume training from ckpt
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g["generator"])
        mpd.load_state_dict(state_dict_do["mpd"])
        msd.load_state_dict(state_dict_do["msd"])
        steps = state_dict_do["steps"] + 1
        last_epoch = state_dict_do["epoch"]
        logger.info(f"Restored checkpoint from {cp_g} and {cp_do}")

    optim_g = torch.optim.AdamW(
        generator.parameters(),
        config.adamw.learning_rate,
        betas=[config.adamw.adam_b1, config.adamw.adam_b2],
    )
    optim_d = torch.optim.AdamW(
        itertools.chain(msd.parameters(), mpd.parameters()),
        config.adamw.learning_rate,
        betas=[config.adamw.adam_b1, config.adamw.adam_b2],
    )

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do["optim_g"])
        optim_d.load_state_dict(state_dict_do["optim_d"])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=config.adamw.lr_decay, last_epoch=last_epoch
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=config.adamw.lr_decay, last_epoch=last_epoch
    )
    scaler_g = GradScaler(device, enabled=config.fp16)
    scaler_d = GradScaler(device, enabled=config.fp16)

    train_loader = create_dataloader(config.train_file, config)
    valid_loader = create_dataloader(config.valid_file, config, shuffle=False)

    melspec = LogMelSpectrogram(
        config.mel.n_fft,
        config.mel.num_mels,
        config.sample_rate,
        config.hifigan.hop_size,
        config.mel.win_size,
        config.mel.fmin,
        config.mel.fmax,
    ).to(device)

    sw = SummaryWriter(config.ckpt_path)
    generator.train()
    mpd.train()
    msd.train()

    for epoch in tqdm(range(max(0, last_epoch), config.training_epochs)):
        start = time.time()

        for batch in train_loader:
            start_b = time.time()
            x = batch.ssl.to(device, non_blocking=True)
            y = batch.audio.to(device, non_blocking=True)
            y_mel = melspec(y.squeeze(1))

            with torch.amp.autocast(enabled=config.fp16, device_type=device):
                y_g_hat = generator(x)
                y_g_hat_mel = melspec(y_g_hat.squeeze(1))

            optim_d.zero_grad()

            with torch.amp.autocast(enabled=config.fp16, device_type=device):
                # MPD
                y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
                loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(
                    y_df_hat_r, y_df_hat_g
                )

                # MSD
                y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
                loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(
                    y_ds_hat_r, y_ds_hat_g
                )

                loss_disc_all = loss_disc_s + loss_disc_f

            scaler_d.scale(loss_disc_all).backward()
            scaler_d.step(optim_d)
            scaler_d.update()

            # Generator
            optim_g.zero_grad()

            with torch.amp.autocast(enabled=config.fp16, device_type=device):
                # L1 Mel-Spectrogram Loss
                loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45

                y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
                y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
                loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
                loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
                loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
                loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
                loss_gen_all = (
                    loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel
                )

            scaler_g.scale(loss_gen_all).backward()
            scaler_g.step(optim_g)
            scaler_g.update()

            # STDOUT logging
            if steps % config.stdout_interval == 0:
                with torch.no_grad():
                    mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()

                logger.info(
                    "Steps : {:,d}, Gen Loss Total : {:4.3f}, Mel-Spec. Error : {:4.3f}, sec/batch : {:4.3f}, peak mem: {:5.2f}GB".format(
                        steps,
                        loss_gen_all,
                        mel_error,
                        time.time() - start_b,
                        torch.cuda.max_memory_allocated() / 1e9,
                    )
                )

            # checkpointing
            if steps % config.checkpoint_interval == 0 and steps != 0:
                ckpt_path = "{}/g_{:08d}.pt".format(config.ckpt_path, steps)
                save_checkpoint(
                    ckpt_path,
                    {"generator": (generator).state_dict()},
                )
                ckpt_path = "{}/do_{:08d}.pt".format(config.ckpt_path, steps)
                save_checkpoint(
                    ckpt_path,
                    {
                        "mpd": (mpd).state_dict(),
                        "msd": (msd).state_dict(),
                        "optim_g": optim_g.state_dict(),
                        "optim_d": optim_d.state_dict(),
                        "steps": steps,
                        "epoch": epoch,
                    },
                )

            # Tensorboard summary logging
            if steps % config.summary_interval == 0:
                sw.add_scalar("training/gen_loss_total", loss_gen_all, steps)
                sw.add_scalar("training/mel_spec_error", mel_error, steps)
                sw.add_scalar("training/disc_loss_total", loss_disc_all, steps)

            # Validation
            if steps % config.validation_interval == 0:
                generator.eval()
                torch.cuda.empty_cache()
                val_err_tot = 0
                with torch.no_grad():
                    for j, batch in enumerate(valid_loader):
                        x = batch.ssl.to(device, non_blocking=True)
                        y = batch.audio.to(device, non_blocking=True)
                        y_mel = melspec(y.squeeze(1))

                        with torch.amp.autocast(
                            enabled=config.fp16, device_type=device
                        ):
                            y_g_hat = generator(x)
                            y_g_hat_mel = melspec(y_g_hat.squeeze(1))

                        y_g_hat_mel = melspec(y_g_hat.squeeze(1))
                        if y_g_hat_mel.shape[-1] != y_mel.shape[-1]:
                            n_pad = config.hop_size
                            y_g_hat = F.pad(y_g_hat, (n_pad // 2, n_pad - n_pad // 2))
                            y_g_hat_mel = melspec(y_g_hat.squeeze(1))

                        val_err_tot += F.l1_loss(y_mel, y_g_hat_mel).item()

                    val_err = val_err_tot / (j + 1)
                    sw.add_scalar("val/mel_spec_error", val_err, steps)
                    logger.info(
                        f"val. done at {steps:,d} steps. mel spec error: {val_err:5.4f}"
                    )

                # go back to training
                generator.train()
                sw.add_scalar(
                    "memory/max_allocated_gb",
                    torch.cuda.max_memory_allocated() / 1e9,
                    steps,
                )
                sw.add_scalar(
                    "memory/max_reserved_gb",
                    torch.cuda.max_memory_reserved() / 1e9,
                    steps,
                )
                if device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.reset_accumulated_memory_stats()

            steps += 1

        scheduler_g.step()
        scheduler_d.step()
        logger.info(
            "Time taken for epoch {} is {} sec".format(
                epoch + 1, int(time.time() - start)
            )
        )


def override_with_args(config: DictConfig, args: dict):
    """
    Update the config with the passed arguments. The new value is casted to the type of
    the existing value. If that fails, an error is raised.
    """

    def check_key_existence(config: DictConfig, key: str, full_key: str):
        """Raise an error if the key does not exist in the cofig."""
        if key not in config:
            raise KeyError(
                f"Subkey '{key}' not found in config for override '{full_key}'"
            )

    if len(args) % 2 != 0:
        raise RuntimeError(
            "The number of config arguments must be even (key-value pairs)"
        )

    # cast args to a dict
    args_dict = dict()
    for key_idx in range(0, len(args), 2):
        args_dict[args[key_idx]] = args[key_idx + 1]

    for key, value in args_dict.items():
        keys = key.split(".")

        # iterate over the keys that serve as directories
        sub_config = config
        for sub_key in keys[:-1]:
            check_key_existence(sub_config, sub_key, key)
            sub_config = sub_config[sub_key]

        # change the value of the last key
        last_key = keys[-1]
        check_key_existence(sub_config, last_key, key)
        if type(sub_config[last_key]) is not type(value):
            old_type = type(sub_config[last_key])
            try:
                value = old_type(value)
            except Exception as e:
                raise ValueError(
                    f"Failed to cast override for '{key}' to {old_type}: {e}"
                )
        sub_config[last_key] = value


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("config")

    # load args and config overrides
    args, config_overrides = parser.parse_known_args()
    config = OmegaConf.load(args.config)
    override_with_args(config, config_overrides)

    # define logging directory
    config.ckpt_path = os.path.join(config.checkpoint_dir, str(int(time.time())))
    os.makedirs(config.ckpt_path)

    # add args to config
    for key, value in vars(args).items():
        config[key] = value

    # dump config with commit hash
    config.commit_hash = (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )
    OmegaConf.save(config, os.path.join(config.ckpt_path, "config.yaml"))

    # create logger
    logger = logging.getLogger("train")
    handler = logging.FileHandler(os.path.join(config.ckpt_path, "train.log"))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.info("Initializing training")

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
        logger.info("Batch size:", config.batch_size)
        config.device = "cuda"
    else:
        logger.info("Batch size set to 1 for CPU")
        config.batch_size = 1
        config.device = "cpu"

    train(config, logger)


if __name__ == "__main__":
    main()
