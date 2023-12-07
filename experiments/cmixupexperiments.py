import argparse
import os
import torch
import json
import numpy as np
from ada.utils.trainutils import set_seed, NpEncoder
from baselines.CMixup.src.data_generate import load_data
from baselines.CMixup.src.algorithm import get_mixup_sample_rate

from ada import get_gammas, clustered_anchor
from ada.utils.train import train
from ada.utils.val import test, cal_worst_acc
from cmixupexperiments_utils import *

def main(dataset: str, method: str, seed: int, alpha:float, anchor_levels: int):

    assert method in ["erm", "cmixup", "manimixup", "mixup", "ada", "localmixup"]
    
    # load config files
    args = argparse.Namespace(**get_config(dataset, method))
    args_dict = vars(args)
    if alpha is not None: args.alpha = alpha
    if anchor_levels is not None: args.anchor_levels = anchor_levels
    
    filename = f'{dataset}_results_{method}_seed{seed}'
    os.makedirs(f"{args.result_root_path}/{dataset}", exist_ok=True)

    result = {
        "seed": seed,
        "batch_size": args_dict["batch_size"], 
        "lr": args_dict["optimiser_args"]["lr"]
    }

    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.cuda = (device == torch.device('cuda'))
    datapacket, ts_data = load_data(args, "cuda" if device == torch.device('cuda') else "cpu")

    set_seed(seed) # need to set seeds twice (see above to reproduce exact results from paper)
    model = get_model(args, ts_data, device)
    if method == "ada":
        result.update({
            "anchor_levels": args_dict["anchor_levels"],
            "use_manifold": args_dict["use_manifold"], 
            "alpha": args_dict["alpha"]
        })
        
        if args.dataset.upper() in ["TimeSeries".upper(), "RCF_MNIST".upper()]: x_y = datapacket["y_train"]
        else: x_y = np.concatenate([datapacket["x_train"],datapacket["y_train"]], axis=1)
        anchors = clustered_anchor(X=x_y, anchor_levels=args_dict["anchor_levels"])
        args_dict["gammas"] = get_gammas(args_dict["alpha"], 20)
        datapacket["anchors"] = {"x_train": anchors}

    elif method in ["cmixup", "mixup", "manimixup"]:
        if args.mixtype == 'kde':
            mixup_idx_sample_rate = get_mixup_sample_rate(args, datapacket, device)
        else:
            mixup_idx_sample_rate = None
        args_dict["mixup_idx_sample_rate"] = mixup_idx_sample_rate
    
    best_mse_model, best_mse = train(args, 
                                     model, 
                                     datapacket, 
                                     ts_data=ts_data, 
                                     is_anchor=(method=="ada"),
                                     is_mixup=(method in ["cmixup", "mixup", "manimixup"]),
                                     is_localmixup=(method=="localmixup"),
                                     verbose=True, 
                                     device=device,
                                     early_stopping=False)
    
    torch.save(best_mse_model.state_dict(), f"{args.result_root_path}/{dataset}/{seed}_model_{method}")
    result_dict_best = test(best_mse_model, datapacket["x_test"], datapacket["y_test"], device)
    if args_dict["is_ood"] == 1:
        result_dict_best = cal_worst_acc(args, datapacket, best_mse_model, result_dict_best, ts_data, device)

    result.update(result_dict_best)

    with open(f'{args.result_root_path}/{dataset}/{filename}.json', 'w') as f:
        json.dump(result, f, cls=NpEncoder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train pose estimation model')
    parser.add_argument('-m', '--method', dest='method', help='Which augmentation method to use. (ada, erm, localmixup, cmixup, mixup, manimixup)', default="ada")
    parser.add_argument('-d', '--dataset', dest='dataset', help='which dataset to use (Airfoil)', default="airfoil")
    parser.add_argument('-s', '--seed', dest='seed', help='which seed to use', default=0, type=int)

    # ada parameters
    parser.add_argument('-a', '--alpha', dest='alpha', help='alpha value for ADA', required=False, type=float)
    parser.add_argument('-k', '--k', dest='k',  help='k value (number of clusters) for ADA', required=False, type=int)

    args = vars(parser.parse_args())

    main(args["dataset"], args["method"], args["seed"], args["alpha"], args["k"])