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

    # module_name: text_recognizer.models
    # class_name: MLP
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
    parser.add_argument("--data_class", type=str, default="MNIST")
    parser.add_argument("--model_class", type=str, default="MLP")
    parser.add_argument("--load_checkpoint", type=str, default=None)

    # 필요한 argument들을 받아오기 위해 data, model을 불러옴
    temp_args, _ = parser.parse_known_args()  # 현재까지 값을 가지고 있는 arg (temp_args)와 없는 arg(_)분리
    data_class = _import_class(f"text_recognizer.data.{temp_args.data_class}")
    model_class = _import_class(f"text_recognizer.models.{temp_args.model_class}")

    # Get data, model, and LitModel specific arguments
    data_group = parser.add_argument_group("Data Args")
    data_class.add_to_argparse(data_group)

    model_group = parser.add_argument_group("Model Args")
    model_class.add_to_argparse(model_group)

    lit_model_group = parser.add_argument_group("LitModel Args")
    lit_models.BaseLitModel.add_to_argparse(lit_model_group)

    parser.add_argument("--help", "-h", action="help")

    return parser