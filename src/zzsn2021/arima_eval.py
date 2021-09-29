import argparse
from math import sqrt
import numpy as np
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from torch_geometric_temporal import temporal_signal_split
from zzsn2021.loaders.Covid19DatasetLoader import Covid19DatasetLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--full_dataset', help='Type of dataset to use (full or small one).', default=False, type=bool)
    parser.add_argument('--horizon', help='The index of day in the future for which to make predictions.', default=1, type=int)
    parser.add_argument('--lags', help='Number of days in the history.', default=12, type=int)

    args = parser.parse_args()
    horizon=args.horizon
    dataset_loader = Covid19DatasetLoader(args.full_dataset, horizon)
    dataset = dataset_loader.get_dataset(lags=args.lags)
    _, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)
    predictions = []
    true_values = []
    for sample in test_dataset:
        for node in range(len(dataset_loader.node_ids)):
            model = ARIMA(np.array(sample.x[node]), order=(5, 1, 0))
            model_fit = model.fit()
            output = model_fit.forecast(steps=horizon)
            predictions.append(output[horizon-1])  # type: ignore
            true_values.append(np.array(sample.y[node]))

    rmse = sqrt(mean_squared_error(true_values, predictions))
    print('Test RMSE: %.3f' % rmse)