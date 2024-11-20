import os
import shutil
from pathlib import Path
import logging
from typing import Dict, List, Union, Tuple, Any, Set
import yaml
from copy import deepcopy
import itertools

def copy_config_to_project(config_path: str, project_dir: Path) -> None:
    """
    프로젝트 디렉토리에 config 파일을 복사합니다.
    
    Args:
        config_path (str): 원본 config 파일의 경로
        project_dir (Path): 프로젝트 디렉토리 경로
    """
    config_filename = os.path.basename(config_path)
    destination = project_dir / config_filename
    shutil.copy2(config_path, destination)
    logging.info(f"Config 파일이 {destination}에 복사되었습니다.")

def setup_logging(log_level: int = logging.INFO) -> None:
    """Set up logging configuration."""
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        validate_config(config)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration file: {e}")

def validate_config(config: Dict[str, Any]) -> None:
    """Validate the configuration dictionary."""
    required_keys = ['root', 'data', 'river_network']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required key in config: {key}")
    
    if 'data_dir' not in config['root']:
        raise ValueError("Missing 'data_dir' in config['root']")
    
    if 'project_name' not in config['root']:
        raise ValueError("Missing 'project_name' in config['root']")
    
    if 'start_date' not in config['data'] or 'end_date' not in config['data']:
        raise ValueError("Missing 'start_date' or 'end_date' in config['data']")
    
    if 'weather_variables' not in config['data']:
        raise ValueError("Missing 'weather_variables' in config['data']")

def yaml_gen(base_config: Dict[str, Any], param_ranges: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """
    Generate a list of experiment configurations based on a base config and parameter ranges.
    
    :param base_config: Base configuration dictionary
    :param param_ranges: Dictionary of parameters to vary and their possible values
    :return: List of configuration dictionaries
    """
    # Generate all combinations of parameter values
    param_combinations = list(itertools.product(*param_ranges.values()))
    
    # Generate configurations
    configs = []
    for combo in param_combinations:
        # Create a deep copy of the base config
        config = yaml.safe_load(yaml.dump(base_config))
        
        # Update config with new parameter values
        for param, value in zip(param_ranges.keys(), combo):
            keys = param.split('.')
            current = config
            for key in keys[:-1]:
                current = current.setdefault(key, {})
            current[keys[-1]] = value
        
        # Generate experiment name
        exp_name_parts = [f"{key.split('.')[-1]}_{value}" for key, value in zip(param_ranges.keys(), combo)]
        config['root']['project_name'] = '_'.join(exp_name_parts)
        
        configs.append(config)
    
    return configs

def save_yaml_configs(configs: List[Dict[str, Any]], output_dir: str):
    """
    Save the generated configurations as YAML files.
    
    :param configs: List of configuration dictionaries
    :param output_dir: Directory to save the YAML files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for i, config in enumerate(configs):
        filename = f"{config['root']['project_name']}.yaml"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

