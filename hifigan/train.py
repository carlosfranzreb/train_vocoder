import argparse
import itertools
import json
import os
import time
import logging
import subprocess

import torch
import torch.nn.functional as F
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .mel_utils import LogMelSpectrogram
from .ssl_dataset import get_dataset_filelist, SslDataset
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


def create_dataloader(
    dataset: Dataset, config: AttrDict, shuffle: bool = True
) -> DataLoader:
    return DataLoader(
        dataset,
        num_workers=config.num_workers,
        shuffle=shuffle,
        batch_size=config.batch_size,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )


def train(config: AttrDict, logger: logging.Logger, device: str):

    # init models
    logger.info(f"Device: {device}")
    generator = Generator(config).to(device)
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
        config.learning_rate,
        betas=[config.adam_b1, config.adam_b2],
    )
    optim_d = torch.optim.AdamW(
        itertools.chain(msd.parameters(), mpd.parameters()),
        config.learning_rate,
        betas=[config.adam_b1, config.adam_b2],
    )

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do["optim_g"])
        optim_d.load_state_dict(state_dict_do["optim_d"])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=config.lr_decay, last_epoch=last_epoch
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=config.lr_decay, last_epoch=last_epoch
    )
    if config.fp16:
        scaler_g = GradScaler()
        scaler_d = GradScaler()

    train_df, valid_df = get_dataset_filelist(config)

    trainset = SslDataset(
        train_df,
        config.segment_size,
        config.n_fft,
        config.num_mels,
        config.hop_size,
        config.win_size,
        config.sampling_rate,
        config.fmin,
        config.fmax,
        n_cache_reuse=0,
        shuffle=True,
        fmax_loss=config.fmax_for_loss,
        device=device,
        audio_root_path=config.audio_root_path,
        feat_root_path=config.feature_root_path,
    )
    train_loader = create_dataloader(trainset, config)

    melspec = LogMelSpectrogram(
        config.n_fft,
        config.num_mels,
        config.sampling_rate,
        config.hop_size,
        config.win_size,
        config.fmin,
        config.fmax,
    ).to(device)

    validset = SslDataset(
        valid_df,
        config.segment_size,
        config.n_fft,
        config.num_mels,
        config.hop_size,
        config.win_size,
        config.sampling_rate,
        config.fmin,
        config.fmax,
        False,
        False,
        n_cache_reuse=0,
        fmax_loss=config.fmax_for_loss,
        device=device,
        audio_root_path=config.audio_root_path,
        feat_root_path=config.feature_root_path,
    )
    validation_loader = create_dataloader(validset, config, shuffle=False)

    sw = SummaryWriter(config.ckpt_path)
    generator.train()
    mpd.train()
    msd.train()

    for epoch in tqdm(range(max(0, last_epoch), config.training_epochs)):
        start = time.time()

        for i, batch in enumerate(train_loader):
            start_b = time.time()
            x, y, _, y_mel = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y_mel = y_mel.to(device, non_blocking=True)
            y = y.unsqueeze(1)

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

            if config.fp16:
                scaler_d.scale(loss_disc_all).backward()
                scaler_d.step(optim_d)
                scaler_d.update()
            else:
                loss_disc_all.backward()
                optim_d.step()

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

            if config.fp16:
                scaler_g.scale(loss_gen_all).backward()
                scaler_g.step(optim_g)
                scaler_g.update()
            else:
                loss_gen_all.backward()
                optim_g.step()

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
                    for j, batch in enumerate(validation_loader):
                        x, y, _, y_mel = batch
                        y_g_hat = generator(x.to(device))
                        y_mel = y_mel.to(device, non_blocking=True)
                        y_g_hat_mel = melspec(y_g_hat.squeeze(1))
                        if y_g_hat_mel.shape[-1] != y_mel.shape[-1]:
                            # pad it
                            n_pad = config.hop_size
                            y_g_hat = F.pad(y_g_hat, (n_pad // 2, n_pad - n_pad // 2))
                            y_g_hat_mel = melspec(y_g_hat.squeeze(1))

                        val_err_tot += F.l1_loss(y_mel, y_g_hat_mel).item()

                        if j <= 4:
                            if steps == 0:
                                sw.add_audio(
                                    "gt/y_{}".format(j),
                                    y[0],
                                    steps,
                                    config.sampling_rate,
                                )

                            sw.add_audio(
                                "generated/y_hat_{}".format(j),
                                y_g_hat[0],
                                steps,
                                config.sampling_rate,
                            )

                    val_err = val_err_tot / (j + 1)
                    sw.add_scalar("validation/mel_spec_error", val_err, steps)
                    logger.info(
                        f"validation run complete at {steps:,d} steps. validation mel spec error: {val_err:5.4f}"
                    )

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


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("input_training_file")
    parser.add_argument("input_validation_file")
    parser.add_argument("audio_root_path")
    parser.add_argument("feature_root_path")
    parser.add_argument("config")
    parser.add_argument("--checkpoint_dir", default="cp_hifigan")
    parser.add_argument("--training_epochs", default=1800, type=int)
    parser.add_argument("--stdout_interval", default=5, type=int)
    parser.add_argument("--checkpoint_interval", default=5000, type=int)
    parser.add_argument("--summary_interval", default=25, type=int)
    parser.add_argument("--validation_interval", default=5000, type=int)
    parser.add_argument("--fp16", default=False, type=bool)

    # load args
    args = parser.parse_args()
    with open(args.config) as f:
        data = f.read()

    # define logging directory
    json_config = json.loads(data)
    config = AttrDict(json_config)
    args.ckpt_path = os.path.join(args.checkpoint_dir, str(int(time.time())))

    # add args to config
    for key, value in vars(args).items():
        config[key] = value

    # dump config with commit hash
    config.commit_hash = (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )
    os.makedirs(args.ckpt_path)
    json.dump(config, open(os.path.join(args.ckpt_path, "config.json")))

    # create logger
    logger = logging.getLogger("train")
    handler = logging.FileHandler(os.path.join(args.ckpt_path, "train.log"))
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
        device = "cuda"
    else:
        logger.info("Batch size set to 1 for CPU")
        config.batch_size = 1
        device = "cpu"

    train(config, logger, device)


if __name__ == "__main__":
    main()
