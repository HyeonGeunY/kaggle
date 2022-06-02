import argparse
import pytorch_lightning as pl
import torch
from torchmetrics import F1


OPTIMIZER = "Adam"
LR = 1e-3
LOSS = "BCEWithLogitsLoss"
ONE_CYCLE_TOTAL_STEPS = 100


class BaseLitModel(pl.LightningModule):
    """
    파이토치 모듈로 initialized 되는 일반벅인 pytorch-lightning class
    """
    
    def __init__(self, model, args: argparse.Namespace = None):
        super().__init__()
        self.model = model 
        self.args = vars(args) if args is not None else {}
        
        optimizer = self.args.get("optimizer", OPTIMIZER)
        
        self.optimizer_class = getattr(torch.optim, optimizer)
        
        self.lr = self.args.get("lr", LR)
        loss = self.arg.get("loss", LOSS)
        loss_class = getattr(torch.nn, loss)
        self.loss_fn = loss_class()
        
        self.one_cycle_max_lr = self.args.get("one_cycle_max_lr", None)
        self.one_cycle_total_steps = self.args.get("one_cycle_total_steps", ONE_CYCLE_TOTAL_STEPS)

        self.train_acc = F1(num_classes=self.model.num_classes, average='macro')
        self.val_acc = F1(num_classes=self.model.num_classes, average='macro')
        self.test_acc = F1(num_classes=self.model.num_classes, average='macro')
        
    
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
    
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x) # forward 매소드 호출
        loss = self.loss_fn(logits, y.to(dtype=torch.long))
        self.log("train_loss", loss)
        self.train_acc(logits, y)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)

        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y.to(dtype=torch.long))
        self.log("val_loss", loss, prog_bar=True)
        self.val_acc(logits, y)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)


    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        self.test_acc(logits, y)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)