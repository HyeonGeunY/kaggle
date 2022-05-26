from pathlib import Path
import argparse

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
    

