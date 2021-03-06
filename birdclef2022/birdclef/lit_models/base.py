import argparse
from cProfile import label
import os
import pytorch_lightning as pl
import torch
import torch.nn as nn

from torchmetrics import F1Score
import soundfile as sf

import numpy as np
import json

from birdclef.data.base_data_module import BaseDataModule

DOWNLOADED_DIRNAME = BaseDataModule.data_dirname() / "birdclef-2022"
SCORED_BIRDS_FILENAME = DOWNLOADED_DIRNAME / "scored_birds.json"

AUDIO_TEMP_DIR = BaseDataModule.data_dirname() / "audio_tmp"

try:
    import wandb
except ModuleNotFoundError:
    pass


OPTIMIZER = "Adam"
LR = 1e-3
LOSS = "BCEWithLogitsLoss"
ONE_CYCLE_TOTAL_STEPS = 100
THRES = 0.3


class BaseLitModel(pl.LightningModule):
    """
    파이토치 모듈로 initialized 되는 일반벅인 pytorch-lightning class
    """

    def __init__(self, model, args: argparse.Namespace = None):
        super().__init__()
        self.model = model
        self.args = vars(args) if args is not None else {}
        self.mapping = self.model.data_config["mapping"]
        self.sr = self.model.data_config["sampling_rate"]

        optimizer = self.args.get("optimizer", OPTIMIZER)

        self.optimizer_class = getattr(torch.optim, optimizer)

        self.lr = self.args.get("lr", LR)
        loss = self.args.get("loss", LOSS)
        loss_class = getattr(torch.nn, loss)
        self.loss_fn = loss_class()

        self.threshold = float(self.args.get("threshold", THRES))

        self.one_cycle_max_lr = self.args.get("one_cycle_max_lr", None)
        self.one_cycle_total_steps = self.args.get("one_cycle_total_steps", ONE_CYCLE_TOTAL_STEPS)
        
        with open(SCORED_BIRDS_FILENAME) as f:
            self.scored_birds = json.load(f)

        self.train_acc = F1Score(
            num_classes=self.model.num_classes, average="weighted", threshold=self.threshold
        )
        self.val_acc = F1Score(
            num_classes=self.model.num_classes, average="weighted", threshold=self.threshold
        )
        self.test_acc = F1Score(
            num_classes=self.model.num_classes, average="weighted", threshold=self.threshold
        )

        self.out_sigmoid = nn.Sigmoid()

        try:
            score_columns = []
            exist_pred_columns = []
            exist_label_columns = []
            for m in self.mapping:
                score_columns.append(f"{m}_score")
                exist_pred_columns.append(f"{m}_exist_pred")
                exist_label_columns.append(f"{m}_exist_label")
            table_columns = ['audio', 'mel_spec', 'predict', 'label'] + score_columns + exist_pred_columns + exist_label_columns
        #     # self.wandb_table_valid = wandb.Table(columns=['Audio', 'mel_spec', 'predict', 'label'])
        #     # self.wandb_table_test = wandb.Table(columns=['Audio', 'mel_spec', 'predict', 'label'])
        #    self.wandb_table_valid = wandb.Table(columns=['mel_spec', 'predict', 'label'])
            # self.wandb_table_test_v1 = wandb.Table(columns=['mel_spec', 'predict', 'label'])
            self.wandb_table_test = wandb.Table(columns=table_columns)
        
        except AttributeError:
            pass

        if not os.path.exists(AUDIO_TEMP_DIR):
            os.makedirs(AUDIO_TEMP_DIR)

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument(
            "--optimizer", type=str, default=OPTIMIZER, help="optimizer class from torch.optim"
        )
        parser.add_argument("--lr", type=float, default=LR)
        parser.add_argument("--one_cycle_max_lr", type=float, default=None)
        parser.add_argument("--one_cycle_total_steps", type=int, default=ONE_CYCLE_TOTAL_STEPS)
        parser.add_argument(
            "--loss", type=str, default=LOSS, help="loss function from torch.nn.functional"
        )
        parser.add_argument("--threshold", type=str, default=THRES, help="새가 존재한다고 정할 기준 확률")
        return parser

    def configure_optimizers(self):
        """
        설정 값으로 optimizer를 구성하여 반환
        returns
        ----------
        if onecycle learning rate
            dict{optimizer, learnrate schduler, monitor}
                monitor : 감독할 loss 종류 (str)
        else
            optimizer
        """
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        if self.one_cycle_max_lr is None:
            # one_cycle 정책을 사용하지 않는다면 고정된 lr을 갖는 optimizer 반환
            return optimizer

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=self.one_cycle_max_lr,
            total_steps=self.one_cycle_total_steps,
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)  # forward 매소드 호출
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss)
        self.train_acc(logits, y.to(dtype=torch.int32))
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        self.val_acc(logits, y.to(dtype=torch.int32))
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

        outputs_valid = self.out_sigmoid(logits.detach())

        for i in range(1):
            output_birds = outputs_valid[i] > self.threshold
            
            if output_birds.sum() == 0:
                output_birds[outputs_valid[i].argmax()] = True
                
            pred_birds = ", ".join(
                [self.mapping[j] for j in range(len(output_birds)) if output_birds[j] == True]
            )
            label_birds = ", ".join(
                [self.mapping[j] for j in range(len(output_birds)) if y[i][j] == True]
            )
            
            if not label_birds:
                label_birds = "other_birds"
            # waveform_x = mel_to_waveform(x[i])

            # temp_path = AUDIO_TEMP_DIR / "temp.wav"
            # sf.write(temp_path, waveform_x, Config.sr)
            try:
                #wandb_table_valid = wandb.Table(columns=['mel_spec', 'predict', 'label'])
                # spec = wandb.Image(x[i])
                # self.wandb_table_valid.add_data(audio, spec, pred_birds, label_birds)
                #wandb_table_valid.add_data(spec, pred_birds, label_birds)
                # self.logger.experiment.log({"val_pred_examples": [wandb.Image(x[i], caption=pred_birds]})
                #wandb.log({"val_pred_examples": wandb_table_valid})
                wandb.log({"val_pred": [wandb.Image(x[i], caption=f"pred:\n {pred_birds} \n\n label:\n {label_birds}")]})
                
            except AttributeError:
                pass

    def test_step(self, batch, batch_idx):
        x, y, path = batch
        logits = self(x)
        self.test_acc(logits, y.to(dtype=torch.int32))
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)

        outputs_test = self.out_sigmoid(logits.detach())

        for i in range(len(outputs_test)):
            output_birds = outputs_test[i] > self.threshold
            
            if output_birds.sum() == 0:
                output_birds[outputs_test[i].argmax()] = True
                
            pred_birds = [self.mapping[j] for j in range(len(output_birds)) if output_birds[j] == True]
            label_birds =  [self.mapping[j] for j in range(len(output_birds)) if y[i][j] == True]
            label_exists = [y[i][j] == True for j in range(len(output_birds))]
            
            if not path[i].rsplit("/")[-2] in self.scored_birds:
                continue
            
            # for b in self.scored_birds:
            #     if b in pred_birds or b in label_birds:
            #         b_in_pred_or_label = True
            #         break
            
            # if not b_in_pred_or_label:
            #     continue
            
            # waveform_x = mel_to_waveform(x[i])
            # temp_path = AUDIO_TEMP_DIR / "temp.wav"
            # sf.write(temp_path, waveform_x, Config.sr)
            try:
                # wandb_table_test = wandb.Table(columns=['mel_spec', 'predict', 'label'])
                spec = wandb.Image(x[i])
                audio = wandb.Audio(path[i] + '.wav', self.sr)
                # self.wandb_table_test.add_data(spec, pred_birds, label_birds)
                self.wandb_table_test.add_data(audio, spec, pred_birds, label_birds, *outputs_test[i], *output_birds, *label_exists)
                    
                # self.wandb_table_test.add_data(audio, spec, pred_birds, label_birds)
                # self.logger.experiment.log({"val_pred_examples": [wandb.Image(x[i], caption=pred_birds]})
                #wandb.log({"test_pred_examples": wandb_table_test})
                #wandb.log({"test_pred": [wandb.Image(x[i], caption=f"pred:\n {pred_birds} \n\n label:\n {label_birds}")]})
            except AttributeError:
                pass
