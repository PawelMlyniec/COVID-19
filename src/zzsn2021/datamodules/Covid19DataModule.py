from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torch_geometric.data.batch import Batch
from zzsn2021.datasets.Covid19Dataset import Covid19Dataset


class Covid19DataModule(LightningDataModule):

    def __init__(
            self,
            val_split: float = .2,
            batch_size: int = 16,
            node_features: int = 5,
            shuffle: bool = False,
            full_dataset: bool = False,
            horizon: int = 1
    ):
        super().__init__()
        self.val_split = val_split
        self.batch_size = batch_size
        self.node_features = node_features
        self.shuffle = shuffle
        self.full_dataset = full_dataset
        self.dataset_train = ...
        self.dataset_val = ...
        self.dataset_test = ...
        self.horizon = horizon
        self.node_ids = None

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Optional[str] = None):
        if stage in (None, 'fit'):
            dataset = Covid19Dataset(lags=self.node_features, full_dataset=self.full_dataset, horizon=self.horizon)
            length = len(dataset)
            self.dataset_train, self.dataset_val = random_split(dataset, [length - (int(length * self.val_split)), int(length * self.val_split)])
        elif stage == 'test':
            self.dataset_test = Covid19Dataset(train=False, lags=self.node_features, full_dataset=self.full_dataset, horizon=self.horizon)
            self.node_ids = self.dataset_test.node_ids

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, collate_fn=Batch, shuffle=self.shuffle)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, collate_fn=Batch)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, collate_fn=Batch)

