import torch
import torch.nn.functional as F
import torchaudio


class LogMelSpectrogram(torch.nn.Module):
    def __init__(
        self,
        n_fft,
        num_mels,
        sampling_rate,
        hop_size,
        win_size,
        fmin,
        fmax,
        center=False,
    ):
        super().__init__()
        self.melspctrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sampling_rate,
            n_fft=n_fft,
            win_length=win_size,
            hop_length=hop_size,
            center=center,
            power=1.0,
            norm="slaney",
            onesided=True,
            n_mels=num_mels,
            mel_scale="slaney",
            f_min=fmin,
            f_max=fmax,
        )
        self.n_fft = n_fft
        self.hop_size = hop_size

    def forward(self, wav):
        wav = F.pad(
            wav,
            ((self.n_fft - self.hop_size) // 2, (self.n_fft - self.hop_size) // 2),
            "reflect",
        )
        mel = self.melspctrogram(wav)
        logmel = torch.log(torch.clamp(mel, min=1e-5))
        return logmel


def mel_spectrogram(
    y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False
):
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sampling_rate,
            n_fft=n_fft,
            n_mels=num_mels,
            f_min=fmin,
            f_max=fmax,
        )
        mel_basis[str(fmax) + "_" + str(y.device)] = (
            torch.from_numpy(mel).float().to(y.device)
        )
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    # print("Padding by", int((n_fft - hop_size)/2), y.shape)
    # pre-padding
    n_pad = hop_size - (y.shape[1] % hop_size)
    y = F.pad(y.unsqueeze(1), (0, n_pad), mode="reflect").squeeze(1)
    # print("intermediate:", y.shape)

    y = F.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[str(y.device)],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = spec.abs().clamp_(3e-5)
    # print("Post: ", y.shape, spec.shape)

    spec = torch.matmul(mel_basis[str(fmax) + "_" + str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec
