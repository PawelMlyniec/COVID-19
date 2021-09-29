from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torch_geometric.data.batch import Batch
from zzsn2021.datasets.ChickenPoxDataset import ChickenPoxDataset


class ChickenPoxDataModule(LightningDataModule):

    def __init__(
            self,
            val_split: float = .2,
            batch_size: int = 16,
            node_features: int = 5,
            shuffle: bool = False,
            full_dataset: bool = True,   # always True for this dataset,
            horizon: int = 1
    ):
        super().__init__()
        self.val_split = val_split
        self.batch_size = batch_size
        self.node_features = node_features
        self.shuffle = shuffle
        self.full_dataset = True
        self.dataset_train = ...
        self.dataset_val = ...
        self.dataset_test = ...
        self.horizon = horizon

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Optional[str] = None):
        if stage in (None, 'fit'):
            dataset = ChickenPoxDataset(lags=self.node_features, horizon=self.horizon)
            length = len(dataset)
            self.dataset_train, self.dataset_val = random_split(dataset, [length - (int(length * self.val_split)), int(length * self.val_split)])
        elif stage == 'test':
            self.dataset_test = ChickenPoxDataset(train=False, lags=self.node_features, horizon=self.horizon)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, collate_fn=Batch, shuffle=self.shuffle)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, collate_fn=Batch)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, collate_fn=Batch)

