import torch
import torch.utils.data as data
import os
from pathlib import Path
import pandas as pd

class DynamicGraphDataset(data.Dataset):
    def __init__(self, config, gauge_id,ds_type = "train"):
        self.config = config
        self.data_dir = Path(self.config['root']['preprocessed_dir'])
        # print(self.data_dir)
        self.gauge_id = gauge_id
        self.window_size = self.config['data']['window_size']
        self.pred_len = self.config["model"]["out_timesteps"]
        data = torch.load(os.path.join(self.data_dir, f"{self.gauge_id}.pt"))

        self.node_attrs = data['node_attrs']
        self.edge_index = data['edge_index']
        self.edge_attr = data['edge_attrs']

        self.sim_streamflow_std = data['sim_streamflow_std']
        self.sim_streamflow_mean = data['sim_streamflow_mean']
        self.obs_streamflow_std = data['streamflow_std']
        self.obs_streamflow_mean = data['streamflow_mean']
        self.outlet_idx = data['idx']

        if ds_type == "train":
            self.date_range = pd.date_range(start=config["data"]["train_start"], end = config["data"]["train_end"])
            if data["weather_daymet_normalized"] is not None:
                self.weather = data['weather_daymet_normalized'].nan_to_num(0)[:len(self.date_range)]
            else:
                self.weather = data['weather'].nan_to_num(0)[:len(self.date_range)]
            self.obs_streamflow = data['streamflow'].nan_to_num(0)[:len(self.date_range)]
            self.sim_streamflow = data['sim_streamflow'].nan_to_num(0)[:len(self.date_range)]
        elif ds_type == "eval":
            self.date_range = pd.date_range(start=config["data"]["eval_start"], end = config["data"]["eval_end"])
            if data["weather_daymet_normalized"] is not None:
                self.weather = data['weather_daymet_normalized'].nan_to_num(0)[-len(self.date_range):]
            else:
                self.weather = data['weather'].nan_to_num(0)[-len(self.date_range):]
            self.obs_streamflow = data['streamflow'].nan_to_num(0)[-len(self.date_range):]
            self.sim_streamflow = data['sim_streamflow'].nan_to_num(0)[-len(self.date_range):]
        elif ds_type == "all":
            self.date_range = pd.date_range(start=config["data"]["start_date"], end = config["data"]["end_date"])
            if data["weather_daymet_normalized"] is not None:
                self.weather = data['weather_daymet_normalized'].nan_to_num(0)
                # shape: (time, num_nodes, feautres)
            else:
                self.weather = data['weather'].nan_to_num(0)
                # shape: (time, num_features)
            self.obs_streamflow = data['streamflow'].nan_to_num(0)
            self.sim_streamflow = data['sim_streamflow'].nan_to_num(0)
        else:
            raise ValueError

        self.time_len = self.obs_streamflow.shape[0]
        self.num_nodes = self.node_attrs.shape[0]
        self.weather = torch.tensor(self.weather, dtype=torch.float32)

    def __len__(self):
        return self.time_len - self.window_size - self.pred_len - 1
    
    def get_graph_data(self):
        return {
            'node_attrs': self.node_attrs,
            'edge_index': self.edge_index,
            'edge_attr': self.edge_attr,
            'idx': self.outlet_idx,
            'gauge_id': self.gauge_id,
        }

    def __getitem__(self, idx):
        # print(self.obs_streamflow[idx:idx+self.window_size].shape)
        return {
            'weather': self.weather[idx+self.pred_len:idx+self.window_size+self.pred_len],
            'node_attrs': self.node_attrs,
            'edge_index': self.edge_index,
            'edge_attr': self.edge_attr,
            'obs_streamflow': self.obs_streamflow[idx:idx+self.window_size].to(torch.float32),
            'sim_streamflow': self.sim_streamflow[idx:idx+self.window_size].to(torch.float32),
            'target_obs': self.obs_streamflow[idx+self.window_size:idx+self.window_size+self.pred_len].to(torch.float32),
            'target_sim': self.sim_streamflow[idx+self.window_size:idx+self.window_size+self.pred_len].to(torch.float32),
        }
