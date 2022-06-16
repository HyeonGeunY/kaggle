if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    
import sys

import argparse
from pathlib import Path
from typing import Optional
import json
import numpy as np
import pandas as pd
import os

import tqdm
import toml

import torch

from birdclef.data.base_data_module import (
    BaseDataModule,
    load_and_print_info,
    check_data_downloaded,
)
from birdclef.data.util import BaseDataset, split_dataset
from birdclef.util import normalize_std, get_split_by_bird, copy_split_audio
from birdclef.data.augmentation_v2 import Config, waveform_to_melspec, mel_to_waveform, make_essentials, _save_mel_labels_essentials, oversampling

METADATA_FILENAME = BaseDataModule.data_dirname() / ".." / "data_raw" / "metadata.toml"
ZIPFILE_PATH = BaseDataModule.data_dirname() / "birdclef-2022.zip"
DOWNLOADED_DIRNAME = BaseDataModule.data_dirname() / "birdclef-2022"
META_DATA_FILENAME = DOWNLOADED_DIRNAME / "train_metadata.csv"
AUDIO_DIR = DOWNLOADED_DIRNAME / "train_audio"

PROCESSED_DATA_DIRNAME = (
    BaseDataModule.data_dirname() / "processed" / "birdclef2022" / Config.version
)
SPLIT_FILENAME = PROCESSED_DATA_DIRNAME / "traintest_filename.json"
ESSENTIALS_FILENAME = PROCESSED_DATA_DIRNAME / "birdclef2022.json"
SCORED_BIRDS_FILENAME = DOWNLOADED_DIRNAME / "scored_birds.json"


SAMPLE_RATE = Config.sr
N_FFT = Config.n_fft
N_MELS = Config.n_mels

WIN_LENGTH = Config.win_length
HOP_LENGTH = Config.hop_length

MIN_SEC = Config.min_sec
F_MIN = Config.f_min
F_MAX = Config.f_max

TRAIN_FRAC = 0.8

OS_FRAC = None

class BirdClef2022_v2(BaseDataModule):
    """Birdclef2022 데이터셋 생성

    Args:
        BaseDataModule (_type_): _description_
    """

    def __init__(self, args: argparse.Namespace = None):
        super().__init__(args)
        self.meta_df = pd.read_csv(META_DATA_FILENAME)
        if "test" in Config.version:
            self.meta_df = self.meta_df[:100]
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
        
        self.oversampling = self.args.get("oversampling", False)
        self.os_frac = self.args.get("os_frac", OS_FRAC)

        self.mel_converter = waveform_to_melspec
        self.inverse_mel_converter = mel_to_waveform

        if not os.path.exists(ESSENTIALS_FILENAME):
            self.prepare_data()

        with open(ESSENTIALS_FILENAME) as f:
            essentials = json.load(f)

        with open(SPLIT_FILENAME) as f:
            self.split_names = json.load(f)
        
        self.trainval_meta = pd.read_csv(PROCESSED_DATA_DIRNAME / "trainval" / f"{Config.version}_meta.csv")
        self.trainval_meta["filepath"] = str(PROCESSED_DATA_DIRNAME) + "/" + "trainval" + "/" + self.trainval_meta["filename"]
        self.test_meta = pd.read_csv(PROCESSED_DATA_DIRNAME / "test" / f"{Config.version}_meta.csv")
        self.test_meta["filepath"] = str(PROCESSED_DATA_DIRNAME) + "/" + "test" + "/" + self.test_meta["filename"]

        self.trainval_meta.label = self.trainval_meta.label.map(lambda x: np.array(eval(x)))
        self.test_meta.label = self.test_meta.label.map(lambda x: np.array(eval(x)))

        self.mapping = list(essentials["birds"])
        self.inverse_mapping = {
            v: k for k, v in enumerate(self.mapping)
        }  # inverse_mapping: chr => class

        self.transform = None
        self.dims = tuple(torch.load(self.trainval_meta.filepath[0]).shape)
        self.output_dims = len(self.mapping)


    def prepare_data(self):

        if not os.path.exists(ZIPFILE_PATH):
            metadata = toml.load(METADATA_FILENAME)
            check_data_downloaded(metadata, BaseDataModule.data_dirname())
            return

        if os.path.exists(ESSENTIALS_FILENAME) and os.path.exists(SPLIT_FILENAME):
            return

        if not os.path.exists(PROCESSED_DATA_DIRNAME):
            os.makedirs(PROCESSED_DATA_DIRNAME)
            make_essentials(self.meta_df)
            
        if not os.path.exists(PROCESSED_DATA_DIRNAME / "test"):

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
            
            trainval_meta = self.trainval_meta
            
            if self.oversampling:
                trainval_meta = oversampling(trainval_meta)
            
            data_trainval = BaseDataset(trainval_meta.filepath, trainval_meta.label)
            self.data_train, self.data_val = split_dataset(
                base_dataset=data_trainval, fraction=TRAIN_FRAC, seed=2022
            )

        if stage == "test" or stage is None:

            self.data_test = BaseDataset(self.test_meta.filepath, self.test_meta.label)

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

        x, y, _ = next(iter(self.train_dataloader()))
        xt, yt, _ = next(iter(self.test_dataloader()))

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
        parser.add_argument("--os_frac", type=float, default=OS_FRAC, help="fraction of oversampling")
        parser.add_argument("--oversampling", action="store_true", default=False)

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


if __name__ == "__main__":
    load_and_print_info(BirdClef2022_v2)
