
import argparse
import json
import numpy as np
from typing import List, Optional

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
import os

from ada import ADA, get_gammas
from ada.models.MLPs import MLP
from ada.utils.train import train, augmentationtype
from ada.utils.val import test
from syntheticdataexperiments import ModelDescription
from experiments.realdataexperiments_utils import *
from baselines.CMixup.src.algorithm import get_mixup_sample_rate
import baselines.CMixup.src.algorithm as Cmixup


def fit_ridge_regression(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, ns:List, ada: bool=False, anchor_levels: int = None, gamma: Optional[List]=None):
    results = {}
    for n in ns:
        results[n] = []
    results["mode"] = [augmentationtype.anchor.value if ada else augmentationtype.erm.value]

    for n in ns: 
        X_input = X_train[:n,:]
        y_input = y_train[:n,:]
        if ada:
            X_input, y_input = ADA(X=X_input, y=y_input,generate_anchor_args={"anchor_levels": anchor_levels}).augment(gamma=gamma, return_original_data=True)
        model = Ridge()
        model.fit(X_input, y_input)
        preds = model.predict(X_test)
        results[n].append(mean_squared_error(y_test, preds))
    return results

def fit_mlp(X_train, y_train, X_valid, y_valid, X_test, y_test, ns, mode, args, hidden_units):

    results = {}
    for n in ns:
        results[n] = []
    
    results["mode"] = [mode.value]

    for i, n in enumerate(ns):
        X_input = X_train[:n,:]
        y_input = y_train[:n]        
        model = MLP(X_input.shape[-1], hidden_units=hidden_units, out_units=1)
        
        if mode == augmentationtype.anchor:
            X_input, y_input = ADA(X=X_input, y=y_input, generate_anchor_args={"anchor_levels": 10}).augment(gamma=get_gammas(4, 20), return_original_data=True)
            datapacket = {
                "x_train": X_input, 
                "y_train": y_input,
                "x_valid": X_valid,
                "y_valid": y_valid,
                "x_test": X_test, 
                "y_test": y_test
            }
            best_model, best_mse = train(args, model, datapacket, ts_data=None, verbose=False, device="cuda" if args.cuda else "cpu", early_stopping=True)
            result_dict_val = test(best_model, datapacket["x_valid"], datapacket["y_valid"], device="cuda" if args.cuda else "cpu")
            result_dict_test = test(best_model, datapacket["x_test"], datapacket["y_test"], device="cuda" if args.cuda else "cpu")
            results[n].append(result_dict_test["mse"])

        elif mode == augmentationtype.erm:
           
            datapacket = {
                "x_train": X_input, 
                "y_train": y_input,
                "x_valid": X_valid,
                "y_valid": y_valid,
                "x_test": X_test, 
                "y_test": y_test
            }
            best_model, best_mse = train(args, model, datapacket, ts_data=None, verbose=False, device="cuda" if args.cuda else "cpu", early_stopping=True)
            result_dict_val = test(best_model, datapacket["x_valid"], datapacket["y_valid"], device="cuda" if args.cuda else "cpu")
            result_dict_test = test(best_model, datapacket["x_test"], datapacket["y_test"], device="cuda" if args.cuda else "cpu")
            results[n].append(result_dict_test["mse"])
        
        elif mode == augmentationtype.cmixup:
            # cmixup 
            datapacket = {
                "x_train": X_input, 
                "y_train": y_input,
                "x_valid": X_valid,
                "y_valid": y_valid,
                "x_test": X_test, 
                "y_test": y_test
            }
            mixup_idx_sample_rate = get_mixup_sample_rate(args, datapacket, device="cuda" if args.cuda else "cpu")
            args.mixup_idx_sample_rate = mixup_idx_sample_rate
            best_model, best_mse = Cmixup.train(args, model, datapacket, ts_data=None, mixup_idx_sample_rate=mixup_idx_sample_rate, device="cuda" if args.cuda else "cpu")
            result_dict_val = test(best_model, datapacket["x_valid"], datapacket["y_valid"], device="cuda" if args.cuda else "cpu")
            result_dict_test = test(best_model, datapacket["x_test"], datapacket["y_test"], device="cuda" if args.cuda else "cpu")
            results[n].append(result_dict_test["mse"])

    return results

def main(dataset: str, model_str: str,  ada:bool=True, erm:bool=True, cmixup:bool=True, seeds: List=[], hidden_units: int=40):
    if model_str=="mlp": model = ModelDescription.mlp
    else: model = ModelDescription.reg

    args = housing_config()
    features, target, val_test_size, training_size, ns, result_tag = get_dataset(dataset, args)

    os.makedirs(f"{args.result_root_path}/Housing", exist_ok=True)
    
    gamma = get_gammas(4, 20)

    results = {}
    results["seed"] = []
    results["mode"] = []
    for n in ns:
        results[n] = []

    num_experiments = 0
    for i, random_seed in enumerate(seeds):
        X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=val_test_size, random_state=random_seed, shuffle=True)
        y_train, y_val = y_train.reshape(-1,1), y_val.reshape(-1, 1)
        X_test, X_val, y_test, y_val = train_test_split(X_val, y_val, test_size=int(val_test_size/2), random_state=random_seed, shuffle=True)

        if model == ModelDescription.reg:
            X_train = np.concatenate([X_train, X_val])
            y_train = np.concatenate([y_train, y_val])

        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        if model == ModelDescription.mlp: 
            X_val = scaler.transform(X_val)
            if ada:
                res = fit_mlp(X_train, y_train, X_val, y_val, X_test, y_test, ns, augmentationtype.anchor, args, hidden_units)
                results["mode"] = results["mode"] + res["mode"]
                for n in ns:
                    results[n] = results[n] + res[n]
            if erm: 
                res = fit_mlp(X_train, y_train, X_val, y_val, X_test, y_test, ns, augmentationtype.erm, args, hidden_units)
                results["mode"] = results["mode"] + res["mode"]
                for n in ns:
                    results[n] = results[n] + res[n]
            if cmixup: 
                res = fit_mlp(X_train, y_train, X_val, y_val, X_test, y_test, ns, augmentationtype.cmixup, args, hidden_units)
                results["mode"] = results["mode"] + res["mode"]
                for n in ns:
                    results[n] = results[n] + res[n]
        else:
            if ada: 
                res = fit_ridge_regression(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, ns=ns, ada=True, anchor_levels=10, gamma=gamma)
                results["mode"] = results["mode"] + res["mode"]
                for n in ns:
                    results[n] = results[n] + res[n]
            if erm:
                res = fit_ridge_regression(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, ns=ns, ada=False)
                results["mode"] = results["mode"] + res["mode"]
                for n in ns:
                    results[n] = results[n] + res[n]

        if i == 0: num_experiments = len(results["mode"])
        results["seed"] = results["seed"] + [random_seed for _ in range(num_experiments)]
        

    if len(seeds) == 1: result_tag = result_tag + f"seed{seeds[0]}"
    with open(f'{args.result_root_path}/Housing/{result_tag}__{model_str}_{hidden_units}.json', 'w') as f:
        json.dump(results, f)

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Experiments on real world dataset.')
    parser.add_argument('-d', '--dataset', dest='dataset', help='dataset to use')
    parser.add_argument('-s', '--seedidx', dest='seedidx', help='index of seed ', default=None)

    parser.add_argument('-m', '--model', dest='model', help='model to use (either an MLP or Ridge Regression)', default="mlp")
    parser.add_argument('-u', '--hiddenunits', dest='hiddenunits', help='Number of hiddenunits in Mlp', type=int, default=40)
    parser.add_argument('--ada', dest="ada", action='store_true')
    parser.add_argument('--erm', dest="erm", action='store_true')
    parser.add_argument('--cmixup', dest="cmixup", action='store_true')

    args = vars(parser.parse_args())
    if not args["seedidx"]:
        seeds = [i for i in range(10)]
    else:
        seeds = [int(args["seedidx"])]

    main(dataset=args['dataset'], model_str=args["model"], ada=args["ada"], erm=args["erm"], cmixup=args["cmixup"], seeds = seeds, hidden_units=args["hiddenunits"])