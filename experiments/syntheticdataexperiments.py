import argparse
import os
import json
import numpy as np
import torch
from enum import Enum
from tqdm import tqdm


from ada import ADA, get_gammas, clustered_anchor
from ada.data.linearData import ModelDescription, LinearData
from ada.utils.train import augmentationtype
from ada.utils.trainutils import NpEncoder
from baselines.CMixup.src.utils import set_seed
from baselines.CMixup.src.algorithm import get_mixup_sample_rate
from baselines.Augmentors.BaselineAugmentors import noise_augmentor, cmixup_augmentor

class ExperimentDescription(Enum):
    erm = "ERM"
    ada = "Anchor Data Augmentation"
    vanilla = "Vanilla Augmentation"
    cmixup = "C-Mixup"

def linear_experiment(model_str: str, hidden_units: int=10, run: int=None, ada:bool=True, erm:bool=True, cmixup:bool=True, vanilla:bool=True):
    
    if run is not None: runs = [run]
    else: runs =range(20)
    if model_str=="mlp": 
        model = ModelDescription.mlp
        ns = [10, 20, 40, 160, 320, 1280, 7040]
    else: 
        model = ModelDescription.reg
        ns = [10, 20, 40, 160, 320] #ridge regression converges faster, so no need to also fit for large sample size (takes longer)
    alphas = [10]

    results = {}
    results["method"] = []
    results["alphas"] = []
    results["run"] = []
    for n in ns:
        results[str(n)] = []

    args = get_synthetic_data_config()
    args_dict = vars(args)
    os.makedirs(f"{args.result_root_path}/lineardata", exist_ok=True)
    number_experiments = 0

    for idx, r in tqdm(enumerate(runs)):
        data = LinearData(random_state_train=r*314, size=ns[-1])

        if erm:
            results["method"].append(ExperimentDescription.erm.value)
            results["alphas"].append(0)
        if ada: 
            for j, a in enumerate(alphas):
                results["method"].append(f"{ExperimentDescription.ada.value} (10 augmentations)")
                results["alphas"].append(a)
                results["method"].append(f"{ExperimentDescription.ada.value} (100 augmentations)")
                results["alphas"].append(a)
        if vanilla:
            if model == ModelDescription.mlp:
                results["method"].append(f"{ExperimentDescription.vanilla.value}")
                results["alphas"].append(0)
            else:
                results["method"].append(f"{ExperimentDescription.vanilla.value} (10 augmentations)")
                results["alphas"].append(0)
                results["method"].append(f"{ExperimentDescription.vanilla.value} (100 augmentations)")
                results["alphas"].append(0)
            
        if cmixup:
            if model == ModelDescription.mlp:
                results["method"].append(f"{ExperimentDescription.cmixup.value}")
                results["alphas"].append(0)
            else:
                results["method"].append(f"{ExperimentDescription.cmixup.value} (10 augmentations)")
                results["alphas"].append(0)
                results["method"].append(f"{ExperimentDescription.cmixup.value} (100 augmentations)")
                results["alphas"].append(0)

        if idx == 0: number_experiments = len(results["method"])
        results["run"] = results["run"] + [r for _ in range(number_experiments)]

        # run experiments/ fill results columns for different number of samples
        for i, n in enumerate(ns):
            X = data.X_train[:n,:]
            y = data.y_train[:n,].reshape(-1, 1)           
            args_dict["batch_size"] = np.min([16, n])

            if erm:
                set_seed(r*314)
                mse = data.fit(model, X_train=X, y_train=y, hidden_units=hidden_units, args=args, mode='erm')
                results[str(n)].append(mse)
            
            if ada:
                for j, a in enumerate(alphas):
                    set_seed(r*314)
                    anchors = clustered_anchor(X=X, y=y, anchor_levels=args_dict["anchor_levels"])
                    args_dict["anchors"] = {"x_train": anchors}
                    args_dict["gammas"] = get_gammas(a, 10)
                    
                    if model == ModelDescription.mlp:
                        mse = data.fit(model, X, y, hidden_units, mode=augmentationtype.anchor, args=args)
                        results[str(n)].append(mse)
                        set_seed(r*314)
                        args_dict["gammas"] = get_gammas(a, 100)
                        mse = data.fit(model, X, y, hidden_units, mode=augmentationtype.anchor, args=args)
                        results[str(n)].append(mse)
                    else:
                        for num_augmentations in [10, 100]:
                            set_seed(r*314)
                            X_til, y_til = ADA(X=X, y=y, anchor=anchors).augment(gamma=get_gammas(a, num_augmentations), return_original_data=True, return_list=False)
                            mse = data.fit(model, X_til, y_til)
                            results[str(n)].append(mse)
            
            if vanilla:
                set_seed(r*314)
                if model == ModelDescription.mlp:
                    mse = data.fit(model, X, y, hidden_units, mode=augmentationtype.vanilla, args=args)
                    results[str(n)].append(mse)
                else:
                    for num_augmentations in [10, 100]:
                        set_seed(r*314)
                        X_til, y_til = noise_augmentor(X=X, y=y, k=num_augmentations, adjust_X=args.vanillaaugment_x, std=args.std, std_x=args.std_x)
                        mse = data.fit(model, X_til, y_til, hidden_units, mode="ada", args=args)
                        results[str(n)].append(mse)
                
            if cmixup:
                set_seed(r*314)
                mixup_idx_sample_rate = get_mixup_sample_rate(args, {'x_train': X, 'y_train': y}, 'cpu')
               
                args_dict["mixup_idx_sample_rate"] = mixup_idx_sample_rate
                if model == ModelDescription.mlp:
                    mse = data.fit(model, X, y, hidden_units, mode=augmentationtype.cmixup, args=args)
                    results[str(n)].append(mse)
                else:
                    for num_augmentations in [10, 100]:
                        set_seed(r*314)
                        X_til, y_til = cmixup_augmentor(X=X, y=y, k=num_augmentations, args=args, return_original=True)
                        mse = data.fit(model, X_til, y_til)
                        results[str(n)].append(mse)

    with open(f"{args.result_root_path}/lineardata/SynLinearData_{model_str}_{run}_.json", "w") as f:
        json.dump(results, f,cls=NpEncoder)

def get_synthetic_data_config():
    args = {
        'result_root_path': 'results',
        'dataset': "synthetic",
        'batch_size': 200,
        'batch_type':0,
        'num_epochs': 1000,
        'optimiser_args': {
            'lr': 0.005,
        },
        'metrics': 'rmse',
        'show_process':0, 
        'store_model':1, 
        'cuda': torch.cuda.is_available(),
        'is_ood': 0,
        'use_manifold': 0, 
        # cmixup 
        'mixtype': 'kde',
        'kde_bandwidth': 1.0,
        'kde_type': 'gaussian',
        'mix_alpha': 2,
        #ADA
        'anchor_levels': 8,
        'alpha': 10,
        #vanilla augmentation
        'std': 0.1, 
        'std_x': 0.05, 
        'vanillaaugment_x': False,
    }
    args = argparse.Namespace(**args)
    return args

def main(model: str, hidden_units: int, run: int, ada: bool, erm: bool, cmixup: bool, vanilla: bool):
    linear_experiment(model_str=model, hidden_units=hidden_units, run=run, ada=ada, erm=erm, cmixup=cmixup, vanilla=vanilla)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Experiments on linear synthetic dataset.')
    parser.add_argument('-m', '--model', dest='model', help='model to use (either an MLP or Ridge Regression)', default="mlp")
    parser.add_argument('-u', '--hiddenunits', dest='hiddenunits', help='Number of hiddenunits in Mlp', type=int, default=10)
    parser.add_argument('-r', '--run', dest='run', help='which run, can be none than runs 0...19 used', type=int, default=None)
    parser.add_argument('--ada', dest="ada", action='store_true')
    parser.add_argument('--erm', dest="erm", action='store_true')
    parser.add_argument('--cmixup', dest="cmixup", action='store_true')
    parser.add_argument('--vanilla', dest="vanilla", action='store_true')
    args = vars(parser.parse_args())
    main(model=args["model"], hidden_units=args["hiddenunits"], run=args['run'], ada=args["ada"], erm=args["erm"], cmixup=args["cmixup"], vanilla=args["vanilla"])