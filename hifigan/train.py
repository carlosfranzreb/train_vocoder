import argparse
import itertools
import json
import os
import time
import logging

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.cuda.amp.grad_scaler import GradScaler
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
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
from .utils import (
    AttrDict,
    build_env,
    load_checkpoint,
    save_checkpoint,
    scan_checkpoint,
)

torch.backends.cudnn.benchmark = True

# create logger
LOGGER = logging.getLogger("train")
handler = logging.FileHandler("train.log")
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)


def train(a, h):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = "cuda"
    else:
        device = "cpu"

    # init models
    LOGGER.info(f"Device: {device}")
    generator = Generator(h).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)

    # check if ckpt folder already exists and retrieve checkpoints
    os.makedirs(a.checkpoint_path, exist_ok=True)
    LOGGER.info("checkpoints directory : ", a.checkpoint_path)
    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(a.checkpoint_path, "g_")
        cp_do = scan_checkpoint(a.checkpoint_path, "do_")

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
        LOGGER.info(f"Restored checkpoint from {cp_g} and {cp_do}")

    optim_g = torch.optim.AdamW(
        generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2]
    )
    optim_d = torch.optim.AdamW(
        itertools.chain(msd.parameters(), mpd.parameters()),
        h.learning_rate,
        betas=[h.adam_b1, h.adam_b2],
    )

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do["optim_g"])
        optim_d.load_state_dict(state_dict_do["optim_d"])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=h.lr_decay, last_epoch=last_epoch
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=h.lr_decay, last_epoch=last_epoch
    )
    if a.fp16:
        scaler_g = GradScaler()
        scaler_d = GradScaler()

    train_df, valid_df = get_dataset_filelist(a)

    trainset = SslDataset(
        train_df,
        h.segment_size,
        h.n_fft,
        h.num_mels,
        h.hop_size,
        h.win_size,
        h.sampling_rate,
        h.fmin,
        h.fmax,
        n_cache_reuse=0,
        shuffle=True,
        fmax_loss=h.fmax_for_loss,
        device=device,
        audio_root_path=a.audio_root_path,
        feat_root_path=a.feature_root_path,
    )

    train_loader = DataLoader(
        trainset,
        num_workers=h.num_workers,
        shuffle=False,
        batch_size=h.batch_size,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )

    melspec = LogMelSpectrogram(
        h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax
    ).to(device)

    validset = SslDataset(
        valid_df,
        h.segment_size,
        h.n_fft,
        h.num_mels,
        h.hop_size,
        h.win_size,
        h.sampling_rate,
        h.fmin,
        h.fmax,
        False,
        False,
        n_cache_reuse=0,
        fmax_loss=h.fmax_for_loss,
        device=device,
        audio_root_path=a.audio_root_path,
        feat_root_path=a.feature_root_path,
    )
    validation_loader = DataLoader(
        validset,
        num_workers=1,
        shuffle=False,
        batch_size=1,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )

    sw = SummaryWriter(os.path.join(a.checkpoint_path, "logs"))

    generator.train()
    mpd.train()
    msd.train()

    for epoch in tqdm(range(max(0, last_epoch), a.training_epochs)):
        start = time.time()

        for i, batch in enumerate(train_loader):
            start_b = time.time()
            x, y, _, y_mel = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y_mel = y_mel.to(device, non_blocking=True)
            y = y.unsqueeze(1)

            with torch.amp.autocast(enabled=a.fp16, device_type=device):
                y_g_hat = generator(x)
                y_g_hat_mel = melspec(y_g_hat.squeeze(1))

            optim_d.zero_grad()

            with torch.amp.autocast(enabled=a.fp16, device_type=device):
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

            if a.fp16:
                scaler_d.scale(loss_disc_all).backward()
                scaler_d.step(optim_d)
                scaler_d.update()
            else:
                loss_disc_all.backward()
                optim_d.step()

            # Generator
            optim_g.zero_grad()

            with torch.amp.autocast(enabled=a.fp16, device_type=device):
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

            if a.fp16:
                scaler_g.scale(loss_gen_all).backward()
                scaler_g.step(optim_g)
                scaler_g.update()
            else:
                loss_gen_all.backward()
                optim_g.step()

            # STDOUT logging
            if steps % a.stdout_interval == 0:
                with torch.no_grad():
                    mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()

                LOGGER.info(
                    "Steps : {:,d}, Gen Loss Total : {:4.3f}, Mel-Spec. Error : {:4.3f}, sec/batch : {:4.3f}, peak mem: {:5.2f}GB".format(
                        steps,
                        loss_gen_all,
                        mel_error,
                        time.time() - start_b,
                        torch.cuda.max_memory_allocated() / 1e9,
                    )
                )

            # checkpointing
            if steps % a.checkpoint_interval == 0 and steps != 0:
                checkpoint_path = "{}/g_{:08d}.pt".format(a.checkpoint_path, steps)
                save_checkpoint(
                    checkpoint_path,
                    {"generator": (generator).state_dict()},
                )
                checkpoint_path = "{}/do_{:08d}.pt".format(a.checkpoint_path, steps)
                save_checkpoint(
                    checkpoint_path,
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
            if steps % a.summary_interval == 0:
                sw.add_scalar("training/gen_loss_total", loss_gen_all, steps)
                sw.add_scalar("training/mel_spec_error", mel_error, steps)
                sw.add_scalar("training/disc_loss_total", loss_disc_all, steps)

            # Validation
            if steps % a.validation_interval == 0:
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
                            n_pad = h.hop_size
                            y_g_hat = F.pad(y_g_hat, (n_pad // 2, n_pad - n_pad // 2))
                            y_g_hat_mel = melspec(y_g_hat.squeeze(1))

                        val_err_tot += F.l1_loss(y_mel, y_g_hat_mel).item()

                        if j <= 4:
                            if steps == 0:
                                sw.add_audio(
                                    "gt/y_{}".format(j),
                                    y[0],
                                    steps,
                                    h.sampling_rate,
                                )

                            sw.add_audio(
                                "generated/y_hat_{}".format(j),
                                y_g_hat[0],
                                steps,
                                h.sampling_rate,
                            )

                    val_err = val_err_tot / (j + 1)
                    sw.add_scalar("validation/mel_spec_error", val_err, steps)
                    LOGGER.info(
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

        LOGGER.info(
            "Time taken for epoch {} is {} sec".format(
                epoch + 1, int(time.time() - start)
            )
        )


def main():
    LOGGER.info("Initializing Training Process..")

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

    a = parser.parse_args()
    LOGGER.info(a)
    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    a.checkpoint_path = os.path.join(a.checkpoint_dir, str(int(time.time())))
    build_env(a.config, "config.json", a.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        LOGGER.info("Batch size:", h.batch_size)
    else:
        LOGGER.info("Batch size set to 1 for CPU")
        h.batch_size = 1

    train(a, h)


if __name__ == "__main__":
    main()
