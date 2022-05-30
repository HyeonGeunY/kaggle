from pathlib import Path
import argparse


import pytorch_lightning as pl
from torch.utils.data import ConcatDataset, DataLoader
from typing import Collection, Dict, Optional, Tuple, Union

from birdclef.data.util import BaseDataset

BATCH_SIZE = 128
NUM_WORKERS = 0


def load_and_print_info(data_module_class) -> None:
    """
    데이터 클래스 로드 & 출력
    """

    parser = argparse.ArgumentParser()
    data_module_class.add_to_argparse(parser)
    args = parser.parse_args()
    dataset = data_module_class(args)
    dataset.prepare_data()
    dataset.setup()
    print(dataset)


def check_data_downloaded(metadata: Dict, dl_dirname: Path) -> Path:
    dl_dirname.mkdir(parents=True, exist_ok=True)
    filename = dl_dirname / metadata["filename"]
    if filename.exists():
        return filename

    print("No data exists")
    print(f"Download data from {metadata['url']} to {filename}")

    return filename


class BaseDataModule(pl.LightningDataModule):
    """
    Base DataModule.
    """

    def __init__(self, args: argparse.Namespace = None) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}
        # arg에 있는 변수들을 반환 없을 시 빈 딕셔너리 반환 vars() 사용시 args는 반드시 __dict__를 가지고 있어야 한다.
        # vars() 지역 변수의 리스트를 반환한다. __dict__ 어트리뷰트를 반환한다. (객체의 내부 변수가 저장된 딕셔너리)
        self.batch_size = self.args.get(
            "batch_size", BATCH_SIZE
        )  # dict.get(key, default=None) 딕셔너리 key에 해당하는 value 반환 없을 시 default 값 반환
        self.num_workers = self.args.get("num_workers", NUM_WORKERS)

        # isinstance(인스턴스, 데이터나 클래스 타입) 두번째 인자로는 튜플 형태로 여러 개가 들어갈 수 있음.
        # 하나라도 만족하면 True 값 반환
        self.on_gpu = isinstance(self.args.get("gpus", None), (str, int))

        # 아래 인자들이 subclass에 있는 지 확인
        self.dims: Tuple[int, ...]
        self.output_dims: Tuple[int, ...]
        self.mapping: Collection
        self.data_train: Union[BaseDataset, ConcatDataset]
        self.data_val: Union[BaseDataset, ConcatDataset]
        self.data_test: Union[BaseDataset, ConcatDataset]

    @classmethod
    def data_dirname(cls):
        # Path : 파일경로 객체로 다루는 라이브러리, 문자열을 사용하는 os.path 모듈보다 편리, resolve() : 상대경로를 절대 경로로 변환
        # relative_to() 절대 경로를 상대 경로로 변환
        # parents[] 상위 경로로 이동 [] 0~ 한칸 상위
        return (
            Path(__file__).resolve().parents[2] / "input" 
        )  # 4칸 상위 디렉토리로 이동 __file__ 현재 코드가 담겨있는 파일의 위치

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument(
            "--batch_size", type=int, default=BATCH_SIZE, help="한 스탭에 사용할 samples의 개수"
        )
        parser.add_argument(
            "--num_workers", type=int, default=NUM_WORKERS, help="데이터 로드에 사용할 process의 개수"
        )
        return parser

    def config(self):
        """
        dataset의 중요 세팅들 반환, instantiate models에 전달되는 값
        """

        return {"input_dims": self.dims, "output_dims": self.output_dims, "mapping": self.mapping}

    def prepare_data(self, *args, **kwrags):
        """
        각 데이터마다 구현
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """
        train, val, test 데이터를 나눔.
        """

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )

