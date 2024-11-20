import argparse
from pathlib import Path
import yaml
from engine import Engine
import os
import torch

def parse_args():
    parser = argparse.ArgumentParser(description='GraphWaveNet Train and Evaluate')
    parser.add_argument('--config', type=str, help='Config file path')
    parser.add_argument('--config_dir', type=str, help='Config directory path')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate'], required=True, help='run mode')
    parser.add_argument('--ev', type=bool, required=True, help='ev or not')
    parser.add_argument('--model_path', type=str, help='Continue running?', default = None)
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_engine(config_path, mode, ev, model_path):
    print(f"Start running {config_path}...")
    engine = Engine(config_path)
    if model_path is not None:
        engine.model.load_state_dict(torch.load(model_path)["model_state_dict"])
        print("start training from... ",model_path)
    else:
        print("No model path specified")
    
    if mode == 'train':
        print(f"{config_path}: Training start...")
        engine.train_full(ev)
        print(f"{config_path}: Training end!")
    
    elif mode == 'evaluate':
        print(f"{config_path}: Evaluation start...")
        _, results = engine.evaluate()
        for gauge_id, result in results.items():
            print(f"{config_path}: Gauge ID: {gauge_id}")
            print(f"{config_path}: NSE: {result['NSE']:.4f}")
            print(f"{config_path}: KGE: {result['KGE']:.4f}")
        print(f"{config_path}: Evaluation end!")

def main():
    args = parse_args()
    if args.config:
        config_paths = [args.config]
    elif args.config_dir:
        config_dir = Path(args.config_dir)
        config_paths = [str(f) for f in config_dir.glob('*.yaml')]
    else:
        raise ValueError("You should select --config or --config_dir.")
    
    for config_path in config_paths:
        run_engine(config_path, args.mode, args.ev, args.model_path)

if __name__ == "__main__":
    main()
