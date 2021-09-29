import json
import numpy as np
from six.moves import urllib
from torch_geometric_temporal import StaticGraphTemporalSignal
import os


class Covid19DatasetLoader(object):
    def __init__(self, full_dataset: bool = False, horizon: int = 1):
        if full_dataset:
            local_data_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "covid19_dataset_full.json")
        else:
            local_data_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "covid19_dataset.json")

        if os.path.exists(local_data_path):
            with open(local_data_path) as f:
                self._dataset = json.load(f)
                f.close()
        else:
            self._read_web_data(full_dataset)
        self._mean = self._dataset["mean"]
        self._std = self._dataset["std"]
        self.horizon = horizon
        self.node_ids = [key.split(", ") for key in self._dataset['node_ids'].keys()]

    def _read_web_data(self, full_dataset: bool):
        if full_dataset:
            url = "https://raw.githubusercontent.com/2021L-ZZSN/11-Mlyniec-Belniak/master/data/covid19_dataset_full.json"
        else:
            url = "https://raw.githubusercontent.com/2021L-ZZSN/11-Mlyniec-Belniak/master/data/covid19_dataset.json"
        self._dataset = json.loads(urllib.request.urlopen(url).read())

    def _get_edges(self):
        self._edges = np.array(self._dataset["edges"]).T

    def _get_edge_weights(self):
        self._edge_weights = np.array(self._dataset["edges_weights"])

    def _get_targets_and_features(self):
        stacked_target = np.array(self._dataset["FX"])
        self.features = [stacked_target[i:i+self.lags,:].T for i in range(stacked_target.shape[0]-self.lags-self.horizon+1)]
        self.targets = [stacked_target[i+self.lags+self.horizon-1,:].T for i in range(stacked_target.shape[0]-self.lags-self.horizon+1)]

    def get_mean(self):
        return self._mean

    def get_std(self):
        return self._std

    def get_dataset(self, lags: int=4) -> StaticGraphTemporalSignal:
        self.lags = lags
        self._get_edges()
        self._get_edge_weights()
        self._get_targets_and_features()
        dataset = StaticGraphTemporalSignal(self._edges, self._edge_weights, self.features, self.targets)
        return dataset