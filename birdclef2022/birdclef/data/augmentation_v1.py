import librosa.feature as F
import torch

class Config:
    version = "v1"
    sr = 32000
    n_fft = 4096
    n_mels = 256
    win_length = None
    hop_length = 512
    f_min = 1000
    f_max = 16000
    mel_norm = 'slaney'
    mel_scale = 'htk'
    center = True
    onesided = True
    pad_mode = 'reflect'
    power=2.0
    min_sec = 5


def waveform_to_melspec(waveform):
    mel_spec = F.melspectrogram(y=waveform.numpy(), sr=Config.sr, n_fft=Config.n_fft, 
                     hop_length=Config.hop_length, win_length=Config.win_length, 
                     center=Config.center, pad_mode=Config.pad_mode, power=Config.power)
    
    log_mel_spec = torch.log10(
            torch.tensor(mel_spec).unsqueeze(0)
            + 1e-10
        )
    
    # log_melspec = normalize_std(log_melspec)

    return log_mel_spec


def mel_to_waveform(mel_spec):
    
    mel_spec = 10 ** (mel_spec)
    waveform = F.inverse.mel_to_audio(M=mel_spec.numpy(), sr=Config.sr, n_fft=Config.n_fft, 
                           hop_length=Config.hop_length, win_length=Config.win_length, 
                           center=Config.center, pad_mode=Config.pad_mode, power=Config.power)
    
    return waveform


def normalize_std(spec):
    """_summary_

    Args:
        spec (_type_): _description_

    Returns:
        _type_: _description_
    """
    return (spec - torch.mean(spec)) / torch.std(spec)