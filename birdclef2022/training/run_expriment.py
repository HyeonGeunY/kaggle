if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
   
    
"""
Experiment-running framework
"""
# import sys
# sys.path.append("..")
import argparse
import importlib

import numpy as np
import torch
import pytorch_lightning as pl
import wandb

from birdclef import lit_models

SEED = 2022
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


def _import_class(module_and_class_name: str) -> type:
    """
    Import class from a module, e.g, 'birdclef.models.SimpleCNN'
    
    모듈로 부터 클래스를 임포트 한다.
    모듈 이름과 클래스 이름을 .을 구분으로 하여 입력으로 받은 후 str.rsplit을 이용하여 모듈과 클래스를 분리하여 사용한다.
    importlib을 이용하여 모듈을 임포트 한 후 getattr을 사용하여 클래스를 받는다.
    """

    module_name, class_name = module_and_class_name.rsplit(
        ".", 1
    )  # rsplit : "." 기준으로 오른쪽부터 두번째 인자 숫자 만큼 나눔 if 0 -> 나누지 않음.

    # module_name: birdclef.models
    # class_name: SimpleCNN
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)

    return class_


def _setup_parser():
    """data, model, trainer, etc 관련 argument에 대한 파이썬 Argumentparser setup"""

    parser = argparse.ArgumentParser(add_help=False) 
    
    trainer_parser = pl.Trainer.add_argparse_args(parser) 
    trainer_parser._action_groups[
        1
    ].title = "Trainer Args" 
    
    parser = argparse.ArgumentParser(
        add_help=False, parents=[trainer_parser]
    ) 

    # Basic arguments
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--data_class", type=str, default="BirdClef2022")
    parser.add_argument("--model_class", type=str, default="SimpleCNN")
    parser.add_argument("--load_checkpoint", type=str, default=None)

    # 필요한 argument들을 받아오기 위해 data, model을 불러옴
    temp_args, _ = parser.parse_known_args()  # 현재까지 값을 가지고 있는 arg (temp_args)와 없는 arg(_)분리
    data_class = _import_class(f"birdclef.data.{temp_args.data_class}")
    model_class = _import_class(f"birdclef.models.{temp_args.model_class}")

    # Get data, model, and LitModel specific arguments
    data_group = parser.add_argument_group("Data Args")
    data_class.add_to_argparse(data_group)

    model_group = parser.add_argument_group("Model Args")
    model_class.add_to_argparse(model_group)

    lit_model_group = parser.add_argument_group("LitModel Args")
    lit_models.BaseLitModel.add_to_argparse(lit_model_group)

    parser.add_argument("--help", "-h", action="help")

    return parser



def main():
    """
    Run an experiment
    Sample command
    '''
    python training/run_experiment.py --max_epochs=3 --gpus='0' --num_workers=20 --model_class=MLP --data_class=MNIST
    '''
    
    _setup_parser를 통해 parser를 받은 후 parse_args()를 이용해 터미널로부터 인자를 전달 받음.
    원하는 loss에 맞게 litmodel 설정.
    
    기존 모델에서 load or 새로 만들기.
    
    wandb 설정.
    
    callbacks 설정
    
    pl.Trainer.from_argparse_args()로 args, callbacks, logge, weights_save_path를 전달하여 Traniner 생성
    
    trainer.tune을 이용하여 lit_model(loss)와 data 인스턴스전달 (datamodule)
    
    trainer.fit(lit_model, datamodule=data) 이용하여 훈련
    trainer.test(lit_model, datamodule=data) 이용하여 테스트
    
    """
    
    parser = _setup_parser()
    args = parser.parse_args()
    data_class = _import_class(f"birdclef.data.{args.data_class}")
    model_class = _import_class(f"birdclef.models.{args.model_class}")
    data = data_class(args)
    model = model_class(data_config=data.config(), args=args)
    
    lit_model_class = lit_models.BaseLitModel
    
    if args.load_checkpoint is not None:  # 기존 model를 load해서 사용할 떄
        lit_model = lit_model_class.load_from_checkpoint(
            args.load_checkpoint, args=args, model=model
        )
    else:
        lit_model = lit_model_class(args=args, model=model)
        
        
    logger = pl.loggers.TensorBoardLogger("training/logs")
    
    
    if args.wandb:
        logger = pl.loggers.WandbLogger()
        logger.watch(model)
        logger.log_hyperparams(vars(args))
        
    
    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=10
    )
    model_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename="{epoch:03d}-{val_loss:.3f}-{val_cer:.3f}", monitor="val_loss", mode="min"
    )
    
    callbacks = [early_stopping_callback, model_checkpoint_callback]

    args.weights_summary = "full"  # model의 전체 summary 출력
    
    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=callbacks, logger=logger, weights_save_path="training/logs"
    )
    
    trainer.tune(
        lit_model, datamodule=data
    )
    
    trainer.fit(lit_model, datamodule=data)
    trainer.test(lit_model, datamodule=data)
    if args.wandb:
        wandb.log({"test_pred_examples": lit_model.wandb_table_test})
    
    best_model_path = model_checkpoint_callback.best_model_path
    if best_model_path:
        print(f"Best model saved at: {best_model_path}")
        if args.wandb:
            wandb.save(best_model_path)
            print("Best model also uploaded to W&B")
          
            
if __name__ == "__main__":
    main()