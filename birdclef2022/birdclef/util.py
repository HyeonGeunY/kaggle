import torch
import numpy as np
import shutil
import os
from typing import List
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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
    return (spec - torch.mean(spec)) / torch.std(spec)


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
    idx_one_bird = list(
        df[(df.primary_label.value_counts() == 1)[df.primary_label].values].primary_label.index
    )
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

    for src, filename in zip(meta_df["filepath"], meta_df.filename):
        dst = root_dir + "/" + filename
        if not os.path.exists(dst.rsplit("/", 1)[0]):
            os.makedirs(dst.rsplit("/", 1)[0])
        shutil.copy(src, dst)


## notebook에서 경로 탐색이 안될시
def add_importpath(package_name: str):
    from importlib.util import find_spec

    if find_spec(package_name) is None:
        import sys

        sys.path.append("..")


## conv layer 출력 이미지 크기 계산
def get_output_size_of_cnn(
    h_in,
    w_in,
    kernel_size: List[int],
    stride: List[int],
    padding: List[int],
    pool=0,
    dilation=[1, 1],
):
    """cnn 출력 이미지 크기를 반환한다.
    Args:
        h_in (int): 입력 이미지 높이
        w_in (int): 입력 이미지 너비
        kernel_size (List[int, int]): 커널 크기
        stride (List[int, int]): 스트라이드
        padding (List[int, int]): 패딩
        pool (int, optional): 풀링. Defaults to 0.
        dilation (list, optional): 커널사이의 간격. Defaults to [1, 1].

    Returns:
        int: 출력 이미지 크기(h, w)
    """

    h_out = np.floor(
        (h_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1
    )
    w_out = np.floor(
        (w_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1
    )

    if pool:
        h_out /= pool
        w_out /= pool

    return int(h_out), int(w_out)


def show_heatmap(df, save_dir=None):
    # make heatmap
    heatmap = []

    for idx, row in df.iterrows():
        _, _, species, sec = row.row_id.split("_")
        if sec == "5":
            sec = "05"
        true_or_false = row.target
        heatmap.append([species,sec,true_or_false])
    
    heatmap = pd.DataFrame(heatmap, columns=["species", "sec", "True_or_False"])

    # show heamap
    fig, ax = plt.subplots(figsize=(10,5))
    cmap = sns.color_palette("Blues")
    heatmap = heatmap.pivot("species", "sec", "True_or_False")
    sns.heatmap(heatmap,ax=ax,linecolor='k',lw=1,cmap=cmap)
    plt.title("Prediction result in soundscape_453028782.ogg")
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, "inference_heatmap" + "." + "png"))
    else:
        plt.show()