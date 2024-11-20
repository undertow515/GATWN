import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import GraphWaveNet
from dataset import DynamicGraphDataset
import os
import yaml
import hydroeval as he
from tqdm import tqdm
import numpy as np
import csv
from pathlib import Path
from utils import copy_config_to_project



class Engine:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.config_path = config_path
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = self._build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['train']['learning_rate'])
        self.criterion = nn.MSELoss()
        self.train_datasets, self.val_datasets = self._load_datasets()
        self.train_loader = [DataLoader(tds, batch_size=self.config['train']['batch_size'], shuffle=True) for tds in self.train_datasets]
        self.val_loader = [DataLoader(vds, batch_size=self.config['train']['batch_size']) for vds in self.val_datasets]
        self.best_val_loss = float('inf')
        self.patience = self.config['train']['patience']
        self.counter = 0
        self.project_dir = Path("./runs") / self.config['root']['project_name']
        self.n = 1000 # number of sampling batches
        self.alpha = 0.5
        self.use_only_loss_obs = self.config["train"]["use_only_loss_obs"]
        self.use_only_weather = self.config["train"]["use_only_weather"]
        os.makedirs(self.project_dir, exist_ok = True)
        self.loss_log_path = os.path.join(self.project_dir, 'loss_log.csv')
        self._initialize_loss_log()
        copy_config_to_project(self.config_path, self.project_dir)

        

    def _build_model(self):
        model = GraphWaveNet(
            dynamic_channels=self.config['model']['dynamic_channels'],
            static_channels=self.config['model']['static_channels'],
            out_channels=self.config['model']['out_channels'],
            out_timesteps=self.config['model']['out_timesteps'],
            dilations=self.config['model']['dilations'],
            residual_channels=self.config['model']['residual_channels'],
            dilation_channels=self.config['model']['dilation_channels'],
            skip_channels=self.config['model']['skip_channels'],
            end_channels=self.config['model']['end_channels'],
            dropout=self.config['model']['dropout']
        ).to(self.device)
        print(model)
        return model

    def _load_datasets(self):
        train_datasets = []
        val_datasets = []
        for gauge_id in self.config['train']['train_gauge_ids']:
            train_datasets.append(DynamicGraphDataset(
                self.config,
                gauge_id,
                ds_type="train"
            ))
        for gauge_id in self.config['eval']['val_gauge_ids']:
            val_datasets.append(DynamicGraphDataset(
                self.config,
                gauge_id,
                ds_type="eval"
            ))
        return train_datasets, val_datasets
    
    def _get_batches(self, n=10):
        batches = []
        # graph_data = []
        for loader in self.train_loader:
            for batch in loader:
                batches.append([batch, loader.dataset.get_graph_data()])
        r_batches = torch.randperm(len(batches))
        return [batches[i] for i in r_batches[:n]], r_batches[:n]

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        batches, batch_idxs = self._get_batches(self.n)
        print("batch_idxs", batch_idxs, " selected")
        for batch, graph_data in tqdm(batches, desc="Training"):
            node_attrs = graph_data['node_attrs'].to(self.device)
            edge_index = graph_data['edge_index'].to(self.device)
            edge_attr = graph_data['edge_attr'].to(self.device)
            outlet_idx = graph_data['idx']
            gauge_id = graph_data['gauge_id']

            num_nodes = node_attrs.shape[0]
            self.optimizer.zero_grad()
            obs_streamflow = batch['obs_streamflow'].to(self.device)
            sim_streamflow = batch['sim_streamflow'].to(self.device).unsqueeze(-1)
            target_obs = batch['target_obs'].to(self.device)
            target_sim = batch['target_sim'].to(self.device)
            if batch['weather'].dim() == 3:
                weather = batch['weather'].unsqueeze(2).expand(-1, -1, num_nodes, -1).to(self.device)
            elif batch['weather'].dim() == 4:
                weather = batch['weather'].to(self.device)
            else:
                raise ValueError
            if self.use_only_loss_obs:
                sim_streamflow[:, :, outlet_idx] = obs_streamflow.unsqueeze(-1)
            if self.use_only_weather:
                dynamic = weather
            else:
                dynamic = torch.cat([weather, sim_streamflow], dim=-1)
            output = self.model(dynamic, node_attrs, edge_index, edge_attr)

            if self.use_only_loss_obs:
                loss = self.criterion(output.squeeze(-1)[..., outlet_idx].squeeze(-1).squeeze(-1), target_obs)
            else:
                loss1 = self.criterion(output.squeeze(-1), target_sim)
                loss2 = self.criterion(output.squeeze(-1)[..., outlet_idx].squeeze(-1), target_obs)
                loss = self.alpha * loss1 + (1-self.alpha) * loss2

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            print(f"gauge_id now : {gauge_id}")
        print(total_loss)
        
        return total_loss / (len(batches))

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        results = dict()
        rs = torch.randperm(len(self.val_loader))
        v_loader_selected = [self.val_loader[i] for i in rs[:50]]

        with torch.no_grad():
            for loader in v_loader_selected:
                gd = loader.dataset.get_graph_data()
                node_attrs = gd['node_attrs'].to(self.device)
                edge_index = gd['edge_index'].to(self.device)
                edge_attr = gd['edge_attr'].to(self.device)
                outlet_idx = gd['idx']
                num_nodes = node_attrs.shape[0]
                pred_list = []
                target_list = []

                for batch in loader:
                    sim_streamflow = batch['sim_streamflow'].to(self.device).unsqueeze(-1)
                    obs_streamflow = batch['obs_streamflow'].to(self.device)
                    target_obs = batch['target_obs'].to(self.device)
                    target_sim = batch['target_sim'].to(self.device)
                    # print(target_obs.shape, target_sim.shape)
                    if batch['weather'].dim() == 3: # (batch, time, features)
                        weather = batch['weather'].unsqueeze(2).expand(-1, -1, num_nodes, -1).to(self.device)
                    elif batch['weather'].dim() == 4: # (batch, time, nodes, features)
                        weather = batch['weather'].to(self.device)
                    if self.use_only_loss_obs:
                        sim_streamflow[:, :, outlet_idx] = obs_streamflow.unsqueeze(-1)
                    if self.use_only_weather:
                        dynamic = weather
                    else:
                        dynamic = torch.cat([weather, sim_streamflow], dim=-1)


                    output = self.model(dynamic, node_attrs, edge_index, edge_attr)

                    all_indices = torch.arange(num_nodes, device=self.device)

                    

                    if self.use_only_loss_obs:
                        loss = self.criterion(output.squeeze(-1)[..., outlet_idx].squeeze(-1), target_obs)
                        pred_list.append(output.squeeze(-1)[..., outlet_idx].squeeze(-1).cpu().numpy())
                        target_list.append(target_obs.cpu().numpy())
                        # print(target_obs.shape)
                    else:
                        loss = self.criterion(output.squeeze(-1), target_sim)
                        pred_list.append(output.squeeze(-1).cpu().numpy())
                        target_list.append(target_sim.cpu().numpy())
                    
                    total_loss += loss.item()
                if self.use_only_loss_obs:
                    y_pred = np.concatenate(pred_list, axis=0) * float(loader.dataset.obs_streamflow_std) + float(loader.dataset.obs_streamflow_mean)
                    y_true = np.concatenate(target_list, axis=0) * float(loader.dataset.obs_streamflow_std) + float(loader.dataset.obs_streamflow_mean)
                    nses = []
                    kges = []
                    num_pred = y_true.shape[-1]
                    for i in range(num_pred):
                        nse = he.nse(y_pred[:,i], y_true[:,i])
                        kge, _, _, _ = he.kge(y_pred[:,i], y_true[:,i])
                        nses.append(nse.item())
                        kges.append(kge.item())
                    results[loader.dataset.gauge_id] = dict()
                    results[loader.dataset.gauge_id]['y_pred'] = y_pred
                    results[loader.dataset.gauge_id]['y_true'] = y_true
                    results[loader.dataset.gauge_id]['NSE'] = nses
                    results[loader.dataset.gauge_id]['KGE'] = kges
                    
                else:
                    y_pred = np.concatenate(pred_list, axis=0) * np.array(loader.dataset.sim_streamflow_std) + np.array(loader.dataset.sim_streamflow_mean)
                    y_true = np.concatenate(target_list, axis=0) * np.array(loader.dataset.sim_streamflow_std) + np.array(loader.dataset.sim_streamflow_mean)
                    results[loader.dataset.gauge_id] = dict()
                    results[loader.dataset.gauge_id]['y_pred'] = y_pred
                    results[loader.dataset.gauge_id]['y_true'] = y_true
                    nses = []
                    kges = []
                    num_nodes = y_pred.shape[-1]
                    for i in range(num_nodes):
                        nse = he.nse(y_pred[:,0,i], y_true[:,0,i])
                        kge, _, _, _ = he.kge(y_pred[:,0,i], y_true[:,0,i])
                        nses.append(nse)
                        kges.append(kge.item())
                    results[loader.dataset.gauge_id]['NSE'] = nses
                    results[loader.dataset.gauge_id]['KGE'] = kges
        
        return total_loss / sum(len(loader) for loader in self.val_loader), results

    def _initialize_loss_log(self):
        with open(self.loss_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Train Loss', 'Val Loss'])

    def _log_loss(self, epoch, train_loss, val_loss):
        with open(self.loss_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, train_loss, val_loss])

    def train_full(self, ev=False):
        for epoch in range(self.config['train']['epochs']):
            if epoch < 5:
                self.use_only_loss_obs = False
                self.alpha -= 0.1
            else:
                self.use_only_loss_obs = True
            train_loss = self.train_epoch()
            if ev==True:
                val_loss, results = self.evaluate()
                
                print(f"Epoch {epoch+1}/{self.config['train']['epochs']}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                for gauge_id, result in results.items():
                    print(f"Gauge ID: {gauge_id}, NSE: {result['NSE']}, KGE: {result['KGE']}")
                
                # saving loss log and checkpoint
                self._log_loss(epoch, train_loss, val_loss)
                self._save_checkpoint(epoch, train_loss, val_loss, results)
            else:
                self._log_loss(epoch, train_loss, 0)
                self._save_checkpoint(epoch, train_loss, 0, 0)


    def _save_checkpoint(self, epoch, train_loss, val_loss, results):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'results': results,
        }


        checkpoint_path = self.project_dir / "checkpoint"
        os.makedirs(checkpoint_path, exist_ok = True)
        torch.save(checkpoint, os.path.join(checkpoint_path, f'checkpoint_epoch_{epoch+1}.pt'))
