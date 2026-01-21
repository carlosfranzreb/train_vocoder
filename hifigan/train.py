import argparse
import itertools
import os
import time
import logging
import subprocess

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf


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
from .datamodules import create_dataloader

torch.backends.cudnn.benchmark = True


class Trainer:
    def __init__(self, config: AttrDict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.device = config.device

        # init models
        self.generator = Generator(config.hifigan).to(self.device)
        self.mpd = MultiPeriodDiscriminator().to(self.device)
        self.msd = MultiScaleDiscriminator().to(self.device)

        # check if ckpt folder already exists and retrieve checkpoints
        os.makedirs(config.ckpt_path, exist_ok=True)
        logger.info("checkpoints directory : ", config.ckpt_path)
        if os.path.isdir(config.ckpt_path):
            cp_g = scan_checkpoint(config.ckpt_path, "g_")
            cp_do = scan_checkpoint(config.ckpt_path, "do_")

        # if ckpt folder is new, start training from scratch
        if cp_g is None or cp_do is None:
            self.steps = 0
            state_dict_do = None
            self.last_epoch = -1

        # otherwise, resume training from ckpt
        else:
            state_dict_g = load_checkpoint(cp_g, self.device)
            state_dict_do = load_checkpoint(cp_do, self.device)
            self.generator.load_state_dict(state_dict_g["generator"])
            self.mpd.load_state_dict(state_dict_do["mpd"])
            self.msd.load_state_dict(state_dict_do["msd"])
            self.steps = state_dict_do["steps"] + 1
            self.last_epoch = state_dict_do["epoch"]
            logger.info(f"Restored checkpoint from {cp_g} and {cp_do}")

        self.optim_g = torch.optim.AdamW(
            self.generator.parameters(),
            config.adamw.learning_rate,
            betas=[config.adamw.adam_b1, config.adamw.adam_b2],
        )
        self.optim_d = torch.optim.AdamW(
            itertools.chain(self.msd.parameters(), self.mpd.parameters()),
            config.adamw.learning_rate,
            betas=[config.adamw.adam_b1, config.adamw.adam_b2],
        )

        if state_dict_do is not None:
            self.optim_g.load_state_dict(state_dict_do["optim_g"])
            self.optim_d.load_state_dict(state_dict_do["optim_d"])

        self.scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
            self.optim_g, gamma=config.adamw.lr_decay, last_epoch=self.last_epoch
        )
        self.scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
            self.optim_d, gamma=config.adamw.lr_decay, last_epoch=self.last_epoch
        )
        self.scaler_g = GradScaler(self.device, enabled=config.fp16)
        self.scaler_d = GradScaler(self.device, enabled=config.fp16)

        self.train_loader = create_dataloader(config.train_file, config)
        self.valid_loader = create_dataloader(config.valid_file, config, shuffle=False)

        self.melspec = LogMelSpectrogram(
            config.mel.n_fft,
            config.mel.num_mels,
            config.sample_rate,
            config.hifigan.hop_size,
            config.mel.win_size,
            config.mel.fmin,
            config.mel.fmax,
        ).to(self.device)

        self.tb_logger = SummaryWriter(config.ckpt_path)
        self.train()

    def train(self):
        self.generator.train()
        self.mpd.train()
        self.msd.train()

        for epoch in tqdm(range(max(0, self.last_epoch), self.config.training_epochs)):
            start = time.time()

            for batch in self.train_loader:
                start_b = time.time()
                x = batch.ssl.to(self.device, non_blocking=True)
                y = batch.audio.to(self.device, non_blocking=True)
                y_mel = self.melspec(y.squeeze(1))

                with torch.amp.autocast(
                    enabled=self.config.fp16, device_type=self.device
                ):
                    y_g_hat = self.generator(x)
                    y_g_hat_mel = self.melspec(y_g_hat.squeeze(1))

                self.optim_d.zero_grad()

                with torch.amp.autocast(
                    enabled=self.config.fp16, device_type=self.device
                ):
                    # MPD
                    y_df_hat_r, y_df_hat_g, _, _ = self.mpd(y, y_g_hat.detach())
                    loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(
                        y_df_hat_r, y_df_hat_g
                    )

                    # MSD
                    y_ds_hat_r, y_ds_hat_g, _, _ = self.msd(y, y_g_hat.detach())
                    loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(
                        y_ds_hat_r, y_ds_hat_g
                    )

                    loss_disc_all = loss_disc_s + loss_disc_f

                self.scaler_d.scale(loss_disc_all).backward()
                self.scaler_d.step(self.optim_d)
                self.scaler_d.update()

                # Generator
                self.optim_g.zero_grad()

                with torch.amp.autocast(
                    enabled=self.config.fp16, device_type=self.device
                ):
                    # L1 Mel-Spectrogram Loss
                    loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45

                    y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.mpd(y, y_g_hat)
                    y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.msd(y, y_g_hat)
                    loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
                    loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
                    loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
                    loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
                    loss_gen_all = (
                        loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel
                    )

                self.scaler_g.scale(loss_gen_all).backward()
                self.scaler_g.step(self.optim_g)
                self.scaler_g.update()

                # compute mel error if needed (TODO: merge the two)
                if (
                    self.steps % self.config.summary_interval == 0
                    or self.steps % self.config.stdout_interval == 0
                ):
                    with torch.no_grad():
                        mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()
                else:
                    mel_error = None

                self.store_checkpoint(epoch)
                self.log2txt(mel_error, loss_gen_all, start_b)
                self.log2tb(loss_gen_all, mel_error, loss_disc_all)

                # Validation
                if self.steps % self.config.validation_interval == 0:
                    self.generator.eval()
                    torch.cuda.empty_cache()
                    val_err_tot = 0
                    with torch.no_grad():
                        for j, batch in enumerate(self.valid_loader):
                            x = batch.ssl.to(self.device, non_blocking=True)
                            y = batch.audio.to(self.device, non_blocking=True)
                            y_mel = self.melspec(y.squeeze(1))

                            with torch.amp.autocast(
                                enabled=self.config.fp16, device_type=self.device
                            ):
                                y_g_hat = self.generator(x)
                                y_g_hat_mel = self.melspec(y_g_hat.squeeze(1))

                            y_g_hat_mel = self.melspec(y_g_hat.squeeze(1))
                            if y_g_hat_mel.shape[-1] != y_mel.shape[-1]:
                                n_pad = self.config.hop_size
                                y_g_hat = F.pad(
                                    y_g_hat, (n_pad // 2, n_pad - n_pad // 2)
                                )
                                y_g_hat_mel = self.melspec(y_g_hat.squeeze(1))

                            val_err_tot += F.l1_loss(y_mel, y_g_hat_mel).item()

                        val_err = val_err_tot / (j + 1)
                        self.tb_logger.add_scalar(
                            "val/mel_spec_error", val_err, self.steps
                        )
                        self.logger.info(
                            f"val. done at {self.steps:,d} steps. mel spec error: {val_err:5.4f}"
                        )

                    # go back to training
                    self.generator.train()
                    self.tb_logger.add_scalar(
                        "memory/max_allocated_gb",
                        torch.cuda.max_memory_allocated() / 1e9,
                        self.steps,
                    )
                    self.tb_logger.add_scalar(
                        "memory/max_reserved_gb",
                        torch.cuda.max_memory_reserved() / 1e9,
                        self.steps,
                    )
                    if self.device == "cuda":
                        torch.cuda.reset_peak_memory_stats()
                        torch.cuda.reset_accumulated_memory_stats()

                self.steps += 1

            self.scheduler_g.step()
            self.scheduler_d.step()
            self.logger.info(
                "Time taken for epoch {} is {} sec".format(
                    epoch + 1, int(time.time() - start)
                )
            )

    def log2tb(self, loss_gen_all: Tensor, mel_error: Tensor, loss_disc_all: Tensor):
        # TODO: check input types
        if self.steps % self.config.summary_interval != 0:
            return

        self.tb_logger.add_scalar("training/gen_loss_total", loss_gen_all, self.steps)
        self.tb_logger.add_scalar("training/mel_spec_error", mel_error, self.steps)
        self.tb_logger.add_scalar("training/disc_loss_total", loss_disc_all, self.steps)

    def log2txt(
        self,
        mel_error: float,
        loss_gen_all: Tensor,
        start_batch: float,
    ):
        # TODO: check input types
        if self.steps % self.config.stdout_interval != 0:
            return

        self.logger.info(
            "Steps : {:,d}, Gen Loss Total : {:4.3f}, Mel-Spec. Error : {:4.3f}, sec/batch : {:4.3f}, peak mem: {:5.2f}GB".format(
                self.steps,
                loss_gen_all,
                mel_error,
                time.time() - start_batch,
                torch.cuda.max_memory_allocated() / 1e9,
            )
        )

    def store_checkpoint(self, epoch: int):
        if self.steps % self.config.checkpoint_interval == 0 and self.steps != 0:
            ckpt_path = "{}/g_{:08d}.pt".format(self.config.ckpt_path, self.steps)
            save_checkpoint(
                ckpt_path,
                {"generator": (self.generator).state_dict()},
            )
            ckpt_path = "{}/do_{:08d}.pt".format(self.config.ckpt_path, self.steps)
            save_checkpoint(
                ckpt_path,
                {
                    "mpd": (self.mpd).state_dict(),
                    "msd": (self.msd).state_dict(),
                    "optim_g": self.optim_g.state_dict(),
                    "optim_d": self.optim_d.state_dict(),
                    "steps": self.steps,
                    "epoch": epoch,
                },
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
        config.self.device = "cuda"
    else:
        logger.info("Batch size set to 1 for CPU")
        config.batch_size = 1
        config.device = "cpu"

    Trainer(config, logger)


if __name__ == "__main__":
    main()
