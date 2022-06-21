import sys

import argparse
from pathlib import Path

def load_and_print_info(data_module_class):
    """
    데이터 클래스 로드 & info 출력
    """
    parser = argparse.ArgumentParser()
    data_module_class.add_to_argparse(parser)  # parser에 값 넣은 후
    args = parser.parse_args()  # 반환 받은 args로
    dataset = data_module_class(args)  # data_module_class 인스턴스 생성
    dataset.prepare_data()
    dataset.setup()
    print(dataset)
    

def check_data_downloaded(metadata: Dict, dl_dirname: Path):
    dl_dirname.mkdir(parents=True, exist_ok=True)
    filename = dl_dirname / metadata["filename"]
    if filename.exists():
        return filename

    print("No data exists")
    print(f"Download data from {metadata['url']} to {filename} and unzip")

    return filename

    
    

class BaseDataModule():
    """
    Basedata module
    """
    
    def __init__(self, args: argparse.Namespace = None):
        self.args = vars(args) if args is not None else {}
    
    
    @classmethod
    def data_dirname(cls):
        return Path(__file__).resolve().parents[3] / "input"
    
    
    def prepare_data(self, *args, , **kwargs):
        """
        데이터 다운로드 같은 작업 처리
        """
        pass
    
    def setup(self, stage: Optional[str] = None):
        """
        데이터 전처리, dataloader등 작업 수행
        """
        pass