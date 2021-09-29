import os

import torch
from pytorch_lightning import LightningModule
from torch_geometric_temporal import STConv, TemporalConv
from zzsn2021.configs import Config


class STGCN(LightningModule):

    def __init__(self, cfg: Config):
        super(STGCN, self).__init__()
        self.cfg = cfg
        experiment_cfg = self.cfg.experiment
        self.stconv1 = STConv(num_nodes=experiment_cfg.num_nodes,
                              in_channels=1,
                              hidden_channels=experiment_cfg.hidden_channels,
                              out_channels=experiment_cfg.output_channels,
                              kernel_size=experiment_cfg.temporal_kernel_size,
                              K=experiment_cfg.graph_kernel_size)
        self.stconv2 = STConv(num_nodes=experiment_cfg.num_nodes,
                              in_channels=experiment_cfg.output_channels,
                              hidden_channels=experiment_cfg.hidden_channels,
                              out_channels=experiment_cfg.output_channels,
                              kernel_size=experiment_cfg.temporal_kernel_size,
                              K=experiment_cfg.graph_kernel_size)
        self.temporal_conv = TemporalConv(in_channels=experiment_cfg.output_channels,
                                          out_channels=experiment_cfg.temporal_conv_output_channels,
                                          kernel_size=experiment_cfg.node_features - 4*(experiment_cfg.temporal_kernel_size - 1)) # each STConv block reduces the number of timesteps by 2 * (kernel_size - 1)
        self.linear = torch.nn.Linear(in_features=experiment_cfg.num_nodes * experiment_cfg.temporal_conv_output_channels,
                                      out_features=experiment_cfg.num_nodes)
        torch.autograd.set_detect_anomaly(True)

    def forward(self, x: torch.FloatTensor, edge_indices, edge_weights) -> torch.FloatTensor:
        x = x.to(self.device)
        h1 = self.stconv1(x, edge_indices, edge_weights)
        h2 = self.stconv2(h1, edge_indices, edge_weights)
        o = self.temporal_conv(h2)
        o = o.resize(o.shape[0], o.shape[1] * o.shape[2] * o.shape[3])
        y = self.linear(o)
        y = y.to("cpu")
        return y

    def save_model(self):
        os.makedirs("models", exist_ok=True)
        torch.save(self.state_dict(), os.path.join("models", "model.pt"))

    def load_model(self):
        self.state_dict(torch.load(os.path.join("models", "model.pt")))
