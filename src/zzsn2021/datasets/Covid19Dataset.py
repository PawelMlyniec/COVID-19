from typing import Any

from torch.utils.data import Dataset
from torch_geometric_temporal import temporal_signal_split
from zzsn2021.loaders.Covid19DatasetLoader import Covid19DatasetLoader


class Covid19Dataset(Dataset):
    def __init__(
            self,
            lags: int = 5,
            train: bool = True,
            full_dataset: bool = False,
            horizon: int = 1
    ) -> None:
        super().__init__()

        loader = Covid19DatasetLoader(full_dataset=full_dataset, horizon=horizon)
        self.dataset = loader.get_dataset(lags=lags)
        self.mean = loader.get_mean()
        self.std = loader.get_std()
        train_dataset, test_dataset = temporal_signal_split(self.dataset, train_ratio=0.8)
        if train:
            data = train_dataset
        else:
            data = test_dataset

        self.data = data
        self.node_ids = loader.node_ids

    def __getitem__(self, index: int) -> Any:
        self.data.t = index
        return self.data.__next__()

    def __len__(self) -> int:
        return self.data.snapshot_count
