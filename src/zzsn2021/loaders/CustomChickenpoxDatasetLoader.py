import numpy as np
from torch_geometric_temporal import ChickenpoxDatasetLoader, StaticGraphTemporalSignal


class CustomChickenpoxDatasetLoader(ChickenpoxDatasetLoader):
    """Customized ChickenpoxDatasetLoader. Adds horizon parameter to set the target day.
        For torough documentation please refer to ChickenpoxDatasetLoader class
    """
    def __init__(self, horizon: int = 1):
        super(CustomChickenpoxDatasetLoader, self).__init__()
        self._read_web_data()
        self.horizon = horizon

    def _get_targets_and_features(self):
        stacked_target = np.array(self._dataset["FX"])
        self.features = [stacked_target[i:i+self.lags,:].T for i in range(stacked_target.shape[0]-self.lags-self.horizon+1)]
        self.targets = [stacked_target[i+self.lags+self.horizon-1,:].T for i in range(stacked_target.shape[0]-self.lags-self.horizon+1)]


    def get_dataset(self, lags: int=4) -> StaticGraphTemporalSignal:
        """Returning the Chickenpox Hungary data iterator.

        Args types:
            * **lags** *(int)* - The number of time lags.
        Return types:
            * **dataset** *(StaticGraphTemporalSignal)* - The Chickenpox Hungary dataset.
        """
        self.lags = lags
        self._get_edges()
        self._get_edge_weights()
        self._get_targets_and_features()
        dataset = StaticGraphTemporalSignal(self._edges, self._edge_weights, self.features, self.targets)
        return dataset
