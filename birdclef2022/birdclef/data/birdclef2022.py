import sys

import argparse
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple
import json
import numpy as np
import pandas as pd
import os

import tqdm
import toml

import torch
import torchaudio
import torchaudio.transforms as T

import librosa.feature as F

from birdclef.data.base_data_module import (
    BaseDataModule,
    load_and_print_info,
    check_data_downloaded,
)
from birdclef.data.util import BaseDataset, split_dataset
from birdclef.util import normalize_std, get_split_by_bird, copy_split_audio
from birdclef.data.augmentation_v1 import Config, waveform_to_melspec, mel_to_waveform


METADATA_FILENAME = "/home/skang/Documents/kaggle/bird_clef/data_raw/metadata.toml"
ZIPFILE_PATH = BaseDataModule.data_dirname() / "birdclef-2022.zip"
DOWNLOADED_DIRNAME = BaseDataModule.data_dirname() / "birdclef-2022"
META_DATA_FILENAME = DOWNLOADED_DIRNAME / "train_metadata.csv"
AUDIO_DIR = DOWNLOADED_DIRNAME / "train_audio"
SPLIT_FILENAME = DOWNLOADED_DIRNAME / "traintest_filename.json"

PROCESSED_DATA_DIRNAME = BaseDataModule.data_dirname() / "processed" / "birdclef2022" / Config.version
ESSENTIALS_FILENAME = (
    PROCESSED_DATA_DIRNAME / "birdclef2022.json"
)


SAMPLE_RATE = Config.sr
N_FFT = Config.n_fft
N_MELS = Config.n_mels

WIN_LENGTH = Config.win_length
HOP_LENGTH = Config.hop_length

MIN_SEC = Config.min_sec
F_MIN = Config.f_min
F_MAX = Config.f_max

TRAIN_FRAC = 0.8


class BirdClef2022(BaseDataModule):
    """Birdclef2022 데이터셋 생성

    Args:
        BaseDataModule (_type_): _description_
    """

    def __init__(self, args: argparse.Namespace = None):
        super().__init__(args)
        self.meta_df = pd.read_csv(META_DATA_FILENAME)
        self.meta_df["filepath"] = str(AUDIO_DIR) + "/" + self.meta_df["filename"]

        self.n_mels = self.args.get("n_mels", N_MELS)
        self.n_fft = self.args.get("n_fft", N_FFT)
        self.sr = self.args.get("sample_rate", SAMPLE_RATE)
        self.win_length = self.args.get("win_length", WIN_LENGTH)
        self.hop_length = self.args.get("hop_length", HOP_LENGTH)
        self.min_sec = self.args.get("min_sec", MIN_SEC)
        self.min_sec_proc = self.sr * self.min_sec
        self.f_min = self.args.get("f_min", F_MIN)
        self.f_max = self.args.get("f_max", F_MAX)

        self.mel_converter = waveform_to_melspec
        self.inverse_mel_converter = mel_to_waveform

        if not os.path.exists(ESSENTIALS_FILENAME):
            self.prepare_data()

        with open(ESSENTIALS_FILENAME) as f:
            essentials = json.load(f)

        with open(SPLIT_FILENAME) as f:
            self.split_names = json.load(f)

        self.processed_filepath_trainval = (
            str(PROCESSED_DATA_DIRNAME)
            + "/"
            + "trainval"
            + "/"
            + pd.Series(
                np.arange(
                    len(torch.load(PROCESSED_DATA_DIRNAME / "trainval" / "label_list.pt"))
                ).astype(str)
            )
            + ".pt"
        )
        self.processed_filepath_test = (
            str(PROCESSED_DATA_DIRNAME)
            + "/"
            + "test"
            + "/"
            + pd.Series(
                np.arange(
                    len(torch.load(PROCESSED_DATA_DIRNAME / "test" / "label_list.pt"))
                ).astype(str)
            )
            + ".pt"
        )

        self.labels_trainval = torch.load(PROCESSED_DATA_DIRNAME / "trainval" / "label_list.pt")
        self.labels_test = torch.load(PROCESSED_DATA_DIRNAME / "test" / "label_list.pt")

        self.mapping = list(essentials["birds"])
        self.inverse_mapping = {
            v: k for k, v in enumerate(self.mapping)
        }  # inverse_mapping: chr => class

        self.transform = None
        self.dims = tuple(torch.load(PROCESSED_DATA_DIRNAME / "trainval" / "0.pt").shape)
        self.output_dims = len(self.mapping)


    def prepare_data(self):

        if not os.path.exists(ZIPFILE_PATH):
            metadata = toml.load(METADATA_FILENAME)
            check_data_downloaded(metadata, BaseDataModule.data_dirname())
            return

        if os.path.exists(ESSENTIALS_FILENAME) and os.path.exists(SPLIT_FILENAME):
            return

        if not os.path.exists(DOWNLOADED_DIRNAME / "test"):
            
            if not os.path.exists(PROCESSED_DATA_DIRNAME):
                os.makedirs(PROCESSED_DATA_DIRNAME)
                
            bird_label = list(self.meta_df["primary_label"].unique())
            essentials = {"birds": bird_label}
            
            with open(ESSENTIALS_FILENAME, "w") as f:
                json.dump(essentials, f)
            
            meta_train, meta_test = get_split_by_bird(self.meta_df)

            traintest_filename = {
                "trainval": list(meta_train.filename),
                "test": list(meta_test.filename),
            }

            with open(SPLIT_FILENAME, "w") as f:
                json.dump(traintest_filename, f)

            with open(SPLIT_FILENAME) as f:
                self.split_names = json.load(f)

            for meta, stage in zip([meta_train, meta_test], ["trainval", "test"]):
                copy_split_audio(meta, root_dir=str(DOWNLOADED_DIRNAME), stage=stage)

        for stage in ["trainval", "test"]:
            _save_mel_labels_essentials(
                self.meta_df[self.meta_df.filename.isin(self.split_names[stage])],
                stage,
                self.min_sec_proc,
                self.mel_converter,
                self.sr,
            )

    def setup(self, stage: Optional[str] = None) -> None:

        if stage == "fit" or stage is None:

            data_trainval = BaseDataset(self.processed_filepath_trainval, self.labels_trainval)
            self.data_train, self.data_val = split_dataset(
                base_dataset=data_trainval, fraction=TRAIN_FRAC, seed=2022
            )

        if stage == "test" or stage is None:

            self.data_test = BaseDataset(self.processed_filepath_test, self.labels_test)

    def __repr__(self) -> str:
        """Print info about the dataset."""
        basic = (
            "birdclef2022  Dataset\n"  # pylint: disable=no-member
            f"Num classes: {len(self.mapping)}\n"
            f"Input dims : {self.dims}\n"
            f"Output dims: {self.output_dims}\n"
        )
        if self.data_train is None and self.data_val is None and self.data_test is None:
            return basic

        x, y = next(iter(self.train_dataloader()))
        xt, yt = next(iter(self.test_dataloader()))

        data = (
            f"Train/val/test sizes: {len(self.data_train)}, {len(self.data_val)}, {len(self.data_test)}\n"
            f"Train Batch x stats: {(x.shape, x.dtype, x.min(), x.mean(), x.std(), x.max())}\n"
            f"Train Batch y stats: {(y.shape, y.dtype, y.min(), y.max())}\n"
            f"Test Batch x stats: {(xt.shape, xt.dtype, xt.min(), xt.mean(), xt.std(), xt.max())}\n"
            f"Test Batch y stats: {(yt.shape, yt.dtype, yt.min(), yt.max())}\n"
        )
        return basic + data

    @staticmethod
    def add_to_argparse(parser):
        parser = BaseDataModule.add_to_argparse(parser)
        parser.add_argument("--n_mels", type=int, default=N_MELS, help="n_mels")
        parser.add_argument("--n_fft", type=int, default=N_FFT, help="n_fft")
        parser.add_argument("--sr", type=int, default=SAMPLE_RATE, help="sampling rate")
        parser.add_argument("--win_length", type=int, default=WIN_LENGTH, help="window_length")
        parser.add_argument("--hop_length", type=int, default=HOP_LENGTH, help="hop length")
        parser.add_argument("--min_sec", type=int, default=MIN_SEC, help="음악 샘플을 나눌 간격(5초)")
        parser.add_argument("--f_min", type=int, default=F_MIN, help="min_frequency of mel spec")
        parser.add_argument("--f_max", type=int, default=F_MAX, help="max_frequency of mel spec")

        return parser

    @property
    def form_filepath(self):
        return {filename: str(AUDIO_DIR) + "/" + filename for filename in self.meta_df.filename}

    @property
    def split_by_id(self):
        return {
            self.form_filepath[filename]: "test"
            if filename in self.split_names["test"]
            else "trainval"
            for filename in self.meta_df.filename
        }


def _save_mel_labels_essentials(
    df: pd.DataFrame, stage, min_sec_proc, mel_converter, sample_rate=32000
):
    """audio data를 mel spectrogram으로 변환한 후 5초 간격으로 나누어서 저장.

    Args:
        df (pd.DataFrame): 오디오 파일 metadata
    """    
    
    with open(ESSENTIALS_FILENAME) as f:
        essentials = json.load(f)
    bird_label = essentials['birds']
    
    data_index = 0
    label_list = []

    for i in range(len(df)):
        data_index = _audio_to_mel_label(
            df["filepath"].iloc[i],
            min_sec_proc,
            sample_rate,
            mel_converter,
            stage,
            data_index,
            label_list,
            bird_label,
            [df["primary_label"].iloc[i]] + eval(df["secondary_labels"].iloc[i]),
        )

    torch.save(np.stack(label_list), PROCESSED_DATA_DIRNAME / stage / "label_list.pt")


def _audio_to_mel_label(
    filepath,
    min_sec_proc,
    sample_rate,
    mel_converter,
    stage="trainval",
    data_index=0,
    label_list=[],
    bird_label=[],
    label_file=[],
):
    """오디오 파일을 mel spectrogram으로 변환 후 5초 간격으로 잘라서 저장

    Args:
        filepath (str): 오디오 파일 경로
        min_sec_proc (int): 자를 간격(5초) * sample rate
        sample_rate (int): 1초에 측정한 샘플 수
        mel_converter (torch.transform): mel_converter
        data_index (int, optional): 파일이름(인덱스). Defaults to 0.
        label_list (list, optional): 각 음원 파일 별 label 정보(target). Defaults to [].
        bird_label (list, optional): 전체 타겟 클래스 정보. Defaults to [].
        label_file (list, optional): 각 파일에 들어있는 타겟 정보. Defaults to [].

    Returns:
        _type_: _description_
    """

    label_file_all = np.zeros(len(bird_label))
    for label_file_temp in label_file:
        label_file_all += label_file_temp == bird_label
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

        if not os.path.exists(PROCESSED_DATA_DIRNAME / stage):
            os.makedirs(PROCESSED_DATA_DIRNAME / stage)

        torch.save(log_melspec, PROCESSED_DATA_DIRNAME / stage / (str(data_index) + ".pt"))
        label_list.append(label_file_all)
        data_index += 1

    return data_index


def repeat_crop_waveform(waveform: torch.tensor, min_sec_proc, wav_len) -> torch.tensor:
    """
    정해진 길이보다 오디오가 짧다면 정해진 길이만큼 오디오를 반복한후 자른다.
    
    Args:
        waveform(torch.tensor): 오디오 파일의 waveform
        min_sec : 최소 시간
    """

    if wav_len < min_sec_proc:
        for _ in range(round(min_sec_proc / wav_len)):
            waveform = torch.cat((waveform, waveform[:, 0:wav_len]), 1)
        wav_len = min_sec_proc
        waveform = waveform[:, 0:wav_len]

    return waveform, wav_len


def pad_audio(self, audio):
    """_summary_

    Args:
        audio (_type_): _description_

    Returns:
        _type_: _description_
    """
    pad_length = self.num_samples - audio.shape[0]
    last_dim_padding = (0, pad_length)
    audio = F.pad(audio, last_dim_padding)
    return audio


def crop_audio(self, audio):
    """_summary_

    Args:
        audio (_type_): _description_

    Returns:
        _type_: _description_
    """
    return audio[: self.num_samples]


def to_mono(waveform: torch.tensor):
    """ 다채널 waveform을 mono로 만들어준다.
    
    Args:
        waveform(torch.tensor): waveform (N_channel, samples)
    
    returns:
        torch.tensor: waveform (1, samples)
    
    """
    return torch.mean(waveform, axis=0)


if __name__ == "__main__":
    load_and_print_info(BirdClef2022)
