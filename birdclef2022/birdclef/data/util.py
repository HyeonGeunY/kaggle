from typing import Any, Callable, Dict, Sequence, Tuple, Union
import torch


class BaseDataset(torch.utils.data.Dataset):
    """
    Parameters
    -----------
    data_path
        data_path
    targets
        labels
    transform
        feature 변환
    target_transform
        targe 변환
    """

    def __init__(
        self, data_path, targets, transform: Callable = None, target_transform: Callable = None,
    ) -> None:

        # 훈련 데이터의 샘플의 수 와 타겟의 수가 같은 지 확인
        if len(data_path) != len(targets):
            raise ValueError("Data and targets must be of equal length")

        super().__init__()
        self.data_path = data_path
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        """
        data 길이 반환
        """
        return len(self.data_path)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        transform 진행한 데이터 반환
        Parameters
        -----------
        index
        Returns
        --------
        (datum, target)
        """
        datum = torch.load(self.data_path[index])
        target = self.targets[index]

        # transform을 딕셔너리에 담고 transform phase를 만드는 것으로 변경 가능
        if self.transform is not None:
            # 전처리
            datum = self.transform(datum)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return datum, target


def split_dataset(
    base_dataset: BaseDataset, fraction: float, seed: int
) -> Tuple[BaseDataset, BaseDataset]:
    """
    base_dataset을 두개로 나눔
    1. fraction * base_dataset의 크기
    2. (1-fraction) * base_dataset의 크기
    """

    split_a_size = int(fraction * len(base_dataset))
    split_b_size = len(base_dataset) - split_a_size

    return torch.utils.data.random_split(
        base_dataset, [split_a_size, split_b_size], generator=torch.Generator().manual_seed(seed)
    )
