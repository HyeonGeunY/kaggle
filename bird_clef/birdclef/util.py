import torch
import numpy as np
import shutil
import os
from sklearn.model_selection import StratifiedShuffleSplit


def torch_fix_seed(seed=42):
    # Python random
    torch.random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True
    
    
def normalize_std(spec):
    return (spec- torch.mean(spec))/torch.std(spec)


from sklearn.model_selection import StratifiedShuffleSplit
def get_split_by_bird(df, bird="primary_label", test_size=0.2, random_state=2022):
    """
    훈련 세트와 테스트 세트로 나눔.
    새가 두 세트 중 하나에만 등장.
    Parameters
    -----------
    df: 모든 새와 레이블(pd.Dataframe)
    bird: 새의 이름이 적혀있는 레이블
    test_size: 테스트 세트로 할당할 비율
    param random_state: 랜덤 시드
    """
    idx_one_bird = list(df[(df.primary_label.value_counts() == 1)[df.primary_label].values].primary_label.index)
    _df = df[(df.primary_label.value_counts() != 1)[df.primary_label].values]
    
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    
    splits = splitter.split(_df, _df[bird])
    train_idx, test_idx = next(splits)
    train_idx = np.concatenate([train_idx, idx_one_bird])
    
    return df.iloc[train_idx, :], df.iloc[test_idx, :]


def copy_split_audio(meta_df, root_dir, stage="trainval"):
    
    if stage not in ["trainval", "test"]:
        print("stage must be one of trainval or test")
        return 
    
    root_dir = root_dir + "/" + stage
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
        
    for src, filename in zip(meta_df['filepath'], meta_df.filename):
        dst = root_dir + "/" + filename
        if not os.path.exists(dst.rsplit("/", 1)[0]):
            os.makedirs(dst.rsplit("/", 1)[0])
        shutil.copy(src, dst)


## notebook에서 경로 탐색이 안될시
def add_importpath(package_name: str):
    from importlib.util import find_spec
    if find_spec(package_name) is None:
        import sys
        sys.path.append('..')