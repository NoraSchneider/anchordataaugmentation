import pandas as pd
import argparse
import torch
from sklearn.datasets import fetch_california_housing, load_boston


def get_dataset(dataset, args):
    if dataset.lower() =="boston":
        features, target = load_boston(return_X_y=True)
        val_test_size = 100
        training_size = features.shape[0]-val_test_size
        sizes = [int(0.1*i*training_size) for i in range(1, 11)]
        result_tag = f"BostonHousing_"
    
    elif dataset.lower() =="california".lower():
        housing = fetch_california_housing()
        target = housing["target"]
        features = housing["data"]
        val_test_size = 20234
        training_size = features.shape[0]-val_test_size
        sizes = [int(0.1*i*training_size) for i in range(1, 11)]
        result_tag = "California_"
    else:
        raise Exception("Dataset not supported.")

    return features, target, val_test_size, training_size, sizes, result_tag


  
def housing_config():
    args = {
        'data_dir': "data/",
        'result_root_path': 'results',
        'dataset': "",
        'batch_size': 64,
        'batch_type':0,
        'num_epochs': 1000,
        'optimiser_args': {
            'lr': 0.001,
        },
        'metrics': 'rmse',
        'show_process':0, 
        'store_model':1, 
        'cuda': torch.cuda.is_available(),
        'is_ood': 0,
        'use_manifold': 0,
        # cmixup 
        'mixtype': 'kde',
        'kde_bandwidth': 1.75,
        'kde_type': 'gaussian',
        'mix_alpha': 2,
    }
    args = argparse.Namespace(**args)
    return args
