import librosa.feature as F
import soundfile as sf
import torch
import json
import pandas as pd
import numpy as np
import torchaudio
import torchaudio.transforms as T
import os
import uuid

from birdclef.data.base_data_module import BaseDataModule
from birdclef.data.util import crop_audio, pad_audio, to_mono, repeat_crop_waveform


class Config:
    version = "v3"
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


DOWNLOADED_DIRNAME = BaseDataModule.data_dirname() / "birdclef-2022"
PROCESSED_DATA_DIRNAME = (
    BaseDataModule.data_dirname() / "processed" / "birdclef2022" / Config.version
)
ESSENTIALS_FILENAME = PROCESSED_DATA_DIRNAME / "birdclef2022.json"
SCORED_BIRDS_FILENAME = DOWNLOADED_DIRNAME / "scored_birds.json"

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


def make_essentials(df):
    bird_label = list(df["primary_label"].unique()) + ["others"]
    essentials = {"birds": bird_label}

    with open(ESSENTIALS_FILENAME, "w") as f:
        json.dump(essentials, f)


def _save_mel_labels_essentials(
    df: pd.DataFrame, stage, min_sec_proc, mel_converter, sample_rate=32000
):
    """audio data??? mel spectrogram?????? ????????? ??? 5??? ???????????? ???????????? ??????.

    Args:
        df (pd.DataFrame): ????????? ?????? metadata
    """

    with open(ESSENTIALS_FILENAME) as f:
        essentials = json.load(f)
        
    with open(SCORED_BIRDS_FILENAME) as f:
        scored_birds = json.load(f)
    bird_label = np.array(essentials["birds"])

    meta_df = pd.DataFrame(columns=["filename", "label"])
    
    for i in range(len(df)):
        meta_df = _audio_to_mel_label(
            df["filepath"].iloc[i],
            min_sec_proc,
            sample_rate,
            mel_converter,
            stage,
            bird_label,
            [df["primary_label"].iloc[i]] + eval(df["secondary_labels"].iloc[i]),
            df["primary_label"].iloc[i],
            scored_birds,
            is_test=False,
            meta_df=meta_df
        )
    
    meta_df.to_csv(PROCESSED_DATA_DIRNAME / stage /  f"{Config.version}_meta.csv", index=False)

def _audio_to_mel_label(
    filepath,
    min_sec_proc,
    sample_rate,
    mel_converter,
    stage="trainval",
    bird_label=None,
    label_file=None,
    primary_label=None,
    scored_birds=None,
    mel_list=None,
    is_test=False,
    meta_df=None
):
    """????????? ????????? mel spectrogram?????? ?????? ??? 5??? ???????????? ????????? ??????

    Args:
        filepath (str): ????????? ?????? ??????
        min_sec_proc (int): ?????? ??????(5???) * sample rate
        sample_rate (int): 1?????? ????????? ?????? ???
        mel_converter (torch.transform): mel_converter
        data_index (int, optional): ????????????(?????????). Defaults to 0.
        label_list (list, optional): ??? ?????? ?????? ??? label ??????(target). Defaults to [].
        bird_label (list, optional): ?????? ?????? ????????? ??????. Defaults to [].
        label_file (list, optional): ??? ????????? ???????????? ?????? ??????. Defaults to [].

    Returns:
        _type_: _description_
    """
    if not is_test:
        if not isinstance(bird_label, np.ndarray):
            bird_label = np.array(bird_label)
        
        b_name = primary_label
        label_file_all = np.zeros(len(bird_label))
        for label_file_temp in label_file:
            label_file_all += label_file_temp == bird_label
            
            if not (primary_label in scored_birds) and (label_file_temp in scored_birds):
                b_name = label_file_temp
        
        if label_file_all.sum() == 0:
            label_file_all[-1] = 1
            
        label_file_all = np.clip(label_file_all, 0, 1)

        waveform, sample_rate_file = torchaudio.load(filepath=filepath)

        if sample_rate_file != sample_rate:
            resample = T.Resample(sample_rate_file, sample_rate)
            waveform = resample(waveform)

        wav_len = waveform.shape[1]
        waveform = to_mono(waveform)
        waveform = waveform.reshape(1, wav_len)

        waveform, wav_len = repeat_crop_waveform(waveform, min_sec_proc, wav_len)

        for index in range(int(wav_len / min_sec_proc)):
            log_melspec = mel_converter(
                waveform[0, index * min_sec_proc : index * min_sec_proc + min_sec_proc]
            )

            if not os.path.exists(PROCESSED_DATA_DIRNAME / stage / b_name):
                os.makedirs(PROCESSED_DATA_DIRNAME / stage / b_name)
            
            spec_name = str(uuid.uuid4())
            torch.save(log_melspec, PROCESSED_DATA_DIRNAME / stage / b_name / (spec_name + ".pt"))
            sf.write(PROCESSED_DATA_DIRNAME / stage / b_name / (spec_name + ".wav"), waveform[0, index * min_sec_proc : index * min_sec_proc + min_sec_proc], sample_rate_file)
            row = pd.DataFrame([[b_name + "/" + spec_name + ".pt", str(list(label_file_all))]], columns=["filename", "label"])
            meta_df = pd.concat([meta_df, row], axis=0, ignore_index=True)
            
        return meta_df

    else:
        waveform, sample_rate_file = torchaudio.load(filepath=filepath)
        wav_len = waveform.shape[1]
        waveform = waveform[0,:].reshape(1, wav_len) # stereo->mono mono->mono
        
        waveform, wav_len = repeat_crop_waveform(waveform, min_sec_proc * 12, wav_len)
        
        for index in range(int(wav_len/min_sec_proc)):
            log_melspec = mel_converter(
                waveform[0, index * min_sec_proc : index * min_sec_proc + min_sec_proc]
            )
            
            mel_list.append(log_melspec)
            
        return mel_list
    

def oversampling(df, frac=None):
    """
    ???????????? oversampling?????? ???????????? ???????????? ????????????.
    """
    
    df["bird_name"] = df.filename.map(lambda x: x.split("/")[0])
    
    if frac is None:
        frac = max(len(df.bird_name.unique()), 1)
        frac = 21 / frac 
    
    sampling_num = int(max(df["bird_name"].value_counts()) * frac)
    birds = df.bird_name.unique()
    df_new = df
    for b in birds:
        if b == "others":
            continue
        
        if len(df[df.bird_name == b]) > sampling_num * 3:
            continue
        
        df_new = pd.concat([df_new, df[df.bird_name == b].sample(n=sampling_num, replace=True, ignore_index=True)], ignore_index=True)
        
    print("Before oversampling")
    print(df.bird_name.value_counts())
    print("After oversampling")
    print(df_new.bird_name.value_counts())
    
    return df
    