import argparse
import torch
from ada.models import *
from baselines.CMixup.src.config import dataset_defaults

config = {  
            "Airfoil":{
                "input_dim": 5, 
                "batch_size": 16,
                "num_epochs": 200,
                "optimiser_args": {
                    "lr": 0.01,
                },
                "metrics": "rmse",
                "is_ood": 0,
                # split
                "train_ratio":0.7,
                "valid_ratio":0.1,
                "id_train_val_split":[0.7, 0.1, 0.2],
                #ADA
                "use_manifold": 1,
                "anchor_levels": 8,
                "alpha": 2,
            },
            "NO2":{
                "input_dim": 7, 
                "batch_size": 32,
                "num_epochs": 125,
                "optimiser_args": {
                    "lr": 5e-4,
                },
                "metrics": "rmse",
                "is_ood": 0,
                # split
                "train_ratio":0.7,
                "valid_ratio":0.1,
                "id_train_val_split":[0.7, 0.1, 0.2],
                # mixup
                "mix_alpha": 2.0,
                #ADA
                "use_manifold": 0,
                "anchor_levels": 4,
                "alpha": 3.5,
            },
            "TimeSeries-exchange_rate":{ # ref -> LSTNet
                "batch_size": 64,
                "num_epochs": 100,
                "optimiser_args": {
                    "lr": 5e-4,
                },
                "metrics": "rmse",
                "is_ood": 0,
                # split
                "train_ratio":0.7,
                "valid_ratio":0.1,
                "id_train_val_split":[0.7, 0.1, 0.2],
                # ts
                "hidCNN": 50, # number of CNN hidden units
                "hidRNN": 50, # number of RNN hidden units
                "window": 24*7, # window size
                "CNN_kernel": 6, # the kernel size of the CNN layers
                "highway_window": 24, # The window size of the highway component
                "clip": 10., # gradient clipping
                "dropout": 0.2, # dropout applied to layers (0 = no dropout)
                "horizon": 12, 
                "skip": 24,
                "hidSkip": 5,
                "L1Loss":False,
                "normalize": 2,
                "output_fun":None,
                # ADA
                "use_manifold": 0,
                "anchor_levels": 40,
                "alpha": 1.125,
                "cluster": "kmeans",
            },

            "TimeSeries-electricity":{
                "batch_size": 256,
                "num_epochs": 100,
                "optimiser_args": {
                    "lr": 0.001,#5e-3,
                },
                "metrics": "rmse",
                "is_ood": 0,
                # split
                "train_ratio":0.7,
                "valid_ratio":0.1,
                "id_train_val_split":[0.7, 0.1, 0.2],
                # ts
                "hidCNN": 50, # number of CNN hidden units
                "hidRNN": 50, # number of RNN hidden units
                "window": 24*7, # window size
                "CNN_kernel": 6, # the kernel size of the CNN layers
                "highway_window": 24, # The window size of the highway component
                "clip": 10., # gradient clipping
                "dropout": 0.2, # dropout applied to layers (0 = no dropout)
                "horizon": 24, 
                "skip": 24,
                "hidSkip": 5,
                "L1Loss":False,
                "normalize": 2,
                "output_fun":"Linear",
                # ADA
                "use_manifold": 1,
                "anchor_levels": 40,
                "alpha": 2,
            },

            "RCF_MNIST":{
                "batch_size": 128,
                "num_epochs": 40,
                "optimiser_args": {
                    "lr": 7e-5,
                },
                "metrics": "rmse",
                "is_ood": 1,
                # split
                "train_ratio":0.7,
                "valid_ratio":0.1,
                "id_train_val_split":[0.7, 0.1, 0.2],
                # rcf
                "use_rotate_class": 0, # 0 -> generate degree randomly; 1 -> sample from 60 fix degree levels
                "spurious_ratio": 0.8,
                "construct_color_data": -1, # 1 -> construct new r-mnist with color spurious information; 0 -> do nothing; -1 -> read rc-fmnist
                "construct_no_color_data": 0, # 1 -> construct r-fmnist; 0 -> do nothing; -1 -> read r-fmnist
                "vis_rcf": 0, # visualize generated data
                "all_pos_color": 0, # 1 -> test data has inverse spurious feature; 0 -> test data has spurious feature as train data
                # ADA
                "use_manifold": 0,
                "anchor_levels": 25,
                "alpha": 3,
            },

            "CommunitiesAndCrime":{
                "input_dim": 122,
                "batch_size": 48,
                "num_epochs": 250, #200
                "optimiser_args": {
                    "lr": 1e-4, #0.001
                },
                "metrics": "rmse",
                "is_ood": 1,
                # split
                "train_ratio":0.7,
                "valid_ratio":0.1,
                "id_train_val_split":[0.7, 0.1, 0.2],
                #ADA
                "use_manifold": 1,
                "anchor_levels": 2,
                "alpha": 2.5,
            },

            "SkillCraft":{
                "input_dim": 17,
                "batch_size": 48,
                "num_epochs": 100,
                "optimiser_args": {
                    "lr": 5e-3,
                },
                "metrics": "rmse",
                "is_ood": 1,
                # split
                "train_ratio":0.7,
                "valid_ratio":0.1,
                "id_train_val_split":[0.7, 0.1, 0.2],
                # mixup
                "mix_alpha": 2.0,
                #ADA
                "use_manifold": 0,
                "anchor_levels": 16,
                "alpha": 4,
            },

            "Dti_dg":{
                "batch_size": 32,
                "num_epochs": 20,
                "optimiser_args": {
                    "lr": 1e-4,
                },
                "metrics": "r",
                "is_ood": 1,
                # split
                "train_ratio":0.7,
                "valid_ratio":0.1,
                "id_train_val_split":[0.7, 0.1, 0.2],
                # mixup
                "mix_alpha": 2.0,
                "mixtype" : "kde", 
                "kde_bandwidth": 21,
                "use_manifold": 1, 
                # ADA
                "use_manifold": 1,
                "anchor_levels": 24,
                "alpha": 3,
                # dti
                "read_dataset": 1, # input dataset directly
                "sub_sample_batch_max_num": 100,
                "store_log": 0,
                "task":"domain_generalization",
                "algorithm":"ERM",
                "hparams": "",
                "hparams_seed": 0,
                "trial_seed": 0,
                "test_envs": [6, 7, 8],
                "output_dir": "train_output",
                "holdout_fraction": 0.2,
                "uda_holdout_fraction": 0., # For domain adaptation, % of test to use unlabeled for training
            },
        }

def get_model(args, ts_data, device):
    if args.dataset == "TimeSeries":
        model = Learner_TimeSeries(args=args,data=ts_data).to(device)
    elif args.dataset == "Dti_dg":
        model = Learner_Dti_dg(hparams=None).to(device)
    elif args.dataset.upper() == "RCF_MNIST".upper():
        model = Learner_RCF_MNIST(args=args).to(device)
    else:
        model = Learner(args=args).to(device)
    return model

def get_config(dataset, method):
    config = {
        "result_root_path": "results",
        "dataset": dataset,
        "show_process": 0, 
        "show_setting": 0,
        "batch_type":0,
        "data_dir": f"data/{dataset}/"
    }
    if dataset.startswith("TimeSeries"):
        config["dataset"] = "TimeSeries"
        config["ts_name"] = dataset
        if dataset == "TimeSeries-electricity":
            config["data_dir"] = f"data/electricity/electricity.txt"
        elif dataset == "TimeSeries-exchange_rate":
            config["data_dir"] = f"data/exchange_rate/exchange_rate.txt"

    if method == "ada":
        config.update(get_config_ADA(dataset=dataset))
    elif method == "cmixup":
        config.update(get_cmixup_config(dataset=dataset))
    elif method == "localmixup":
        config.update(get_local_mixup_config(dataset=dataset))
    else:
        config.update(get_mixup_config(dataset=dataset, manimixup=method=="manimixup"))
    return config

def get_config_ADA(dataset):
    return config[dataset]

# equivalent to readme in CMixup repo & hyperparameters in the paper
def get_cmixup_config(dataset):
    config_dict = dataset_defaults[dataset].copy()
    config_dict["kde_type"] = "gaussian"
    if dataset == "Airfoil": 
        config_dict["mix_alpha"] = 0.5
        config_dict["mixtype"] = "kde"
        config_dict["kde_bandwidth"] = 1.75
        config_dict["use_manifold"] = 1

    elif dataset == "NO2":
        config_dict["mix_alpha"] = 0.5
        config_dict["mixtype"] = "kde"
        config_dict["kde_bandwidth"] = 5e-2
        config_dict["use_manifold"] = 1

    elif dataset == "TimeSeries-exchange_rate":
        config_dict["mix_alpha"] = 1.5
        config_dict["mixtype"] =  "kde"
        config_dict["kde_bandwidth"] =  5e-2
        config_dict["use_manifold"] = 1

    elif dataset == "TimeSeries-electricity":
        config_dict["mix_alpha"] = 2.0
        config_dict["mixtype"] =  "random"
        config_dict["kde_bandwidth"] = 0.5
        config_dict["use_manifold"] = 1

    elif dataset == "RCF-MNIST":
        config_dict["mix_alpha"] =  2.0
        config_dict["mixtype"] =  "random"
        config_dict["kde_bandwidth"] =  0.2
        config_dict["use_manifold"] =  1

    elif dataset == "CommunitiesAndCrime":
        config_dict["mix_alpha"] =  2.0
        config_dict["mixtype"] =  "kde",
        config_dict["kde_bandwidth"] = 1.0
        config_dict["use_manifold"] =  1

    elif dataset == "SkillCraft":
        config_dict["mix_alpha"] =  2.0
        config_dict["mixtype"] =  "kde"
        config_dict["kde_bandwidth"] = 1.0
        config_dict["use_manifold"] =  0,

    elif dataset == "Dti_dg":
        config_dict["mix_alpha"] = 2.0
        config_dict["mixtype"] =  "random"
        config_dict["kde_bandwidth"] = 21
        config_dict["use_manifold"] = 0

    return config_dict

def get_local_mixup_config(dataset):

    config = {  
            "Airfoil":{
                "input_dim": 5, 
                "batch_size": 16,
                "num_epochs": 150,
                "optimiser_args": {
                    "lr": 0.01,
                },
                "metrics": "rmse",
                "is_ood": 0,
                # split
                "train_ratio":0.7,
                "valid_ratio":0.1,
                "id_train_val_split":[0.7, 0.1, 0.2],
                #LM
                "use_manifold": 0,
                "eps": 3.5,
            },
            "NO2":{
                "input_dim": 7, 
                "batch_size": 32,
                "num_epochs": 120,
                "optimiser_args": {
                    "lr": 0.01,
                },
                "metrics": "rmse",
                "is_ood": 0,
                # split
                "train_ratio":0.7,
                "valid_ratio":0.1,
                "id_train_val_split":[0.7, 0.1, 0.2],
                #ADA
                "use_manifold": 0,
                "eps": 2.5,
            },
            "TimeSeries-exchange_rate":{ # ref -> LSTNet
                "batch_size": 128,
                "num_epochs": 200,
                "optimiser_args": {
                    "lr": 5e-4,
                },
                "metrics": "rmse",
                "is_ood": 0,
                # split
                "train_ratio":0.7,
                "valid_ratio":0.1,
                "id_train_val_split":[0.7, 0.1, 0.2],
                # ts
                "hidCNN": 50, # number of CNN hidden units
                "hidRNN": 50, # number of RNN hidden units
                "window": 24*7, # window size
                "CNN_kernel": 6, # the kernel size of the CNN layers
                "highway_window": 24, # The window size of the highway component
                "clip": 10., # gradient clipping
                "dropout": 0.2, # dropout applied to layers (0 = no dropout)
                "horizon": 12, 
                "skip": 24,
                "hidSkip": 5,
                "L1Loss":False,
                "normalize": 2,
                "output_fun":None,
                # LocalMixup
                "use_manifold": 0,
                "eps": 2.5,
            },

            "TimeSeries-electricity":{
                "batch_size": 128,
                "num_epochs": 200,
                "optimiser_args": {
                    "lr": 0.001,
                },
                "metrics": "rmse",
                "is_ood": 0,
                # split
                "train_ratio":0.7,
                "valid_ratio":0.1,
                "id_train_val_split":[0.7, 0.1, 0.2],
                # ts
                "hidCNN": 50, # number of CNN hidden units
                "hidRNN": 50, # number of RNN hidden units
                "window": 24*7, # window size
                "CNN_kernel": 6, # the kernel size of the CNN layers
                "highway_window": 24, # The window size of the highway component
                "clip": 10., # gradient clipping
                "dropout": 0.2, # dropout applied to layers (0 = no dropout)
                "horizon": 24, 
                "skip": 24,
                "hidSkip": 5,
                "L1Loss":False,
                "normalize": 2,
                "output_fun":"Linear",
                # LocalMixup
                "use_manifold": 0,
                "eps": 2.5,
            },
            "RCF_MNIST":{
                "batch_size": 64,
                "num_epochs": 30,
                "optimiser_args": {
                    "lr": 0.001,
                },
                "metrics": "rmse",
                "is_ood": 1,
                # split
                "train_ratio":0.7,
                "valid_ratio":0.1,
                "id_train_val_split":[0.7, 0.1, 0.2],
                # rcf
                "use_rotate_class": 0, # 0 -> generate degree randomly; 1 -> sample from 60 fix degree levels
                "spurious_ratio": 0.8,
                "construct_color_data": -1, # 1 -> construct new r-mnist with color spurious information; 0 -> do nothing; -1 -> read rc-fmnist
                "construct_no_color_data": 0, # 1 -> construct r-fmnist; 0 -> do nothing; -1 -> read r-fmnist
                "vis_rcf": 0, # visualize generated data
                "all_pos_color": 0, # 1 -> test data has inverse spurious feature; 0 -> test data has spurious feature as train data
                # LM
                "use_manifold": 0,
                "eps": 1
            },

            "CommunitiesAndCrime":{
                "input_dim": 122,
                "batch_size": 48,
                "num_epochs": 250,
                "optimiser_args": {
                    "lr": 1e-4,
                },
                "metrics": "rmse",
                "is_ood": 1,
                # split
                "train_ratio":0.7,
                "valid_ratio":0.1,
                "id_train_val_split":[0.7, 0.1, 0.2],
                # LM
                "use_manifold": 0,
                "eps": 1
            },

            "SkillCraft":{
                "input_dim": 17,
                "batch_size": 48,
                "num_epochs": 150,
                "optimiser_args": {
                    "lr": 0.001,
                },
                "metrics": "rmse",
                "is_ood": 1,
                # split
                "train_ratio":0.7,
                "valid_ratio":0.1,
                "id_train_val_split":[0.7, 0.1, 0.2],
                # LM
                "use_manifold": 0,
                "eps": 1
            },

            "Dti_dg":{
                "batch_size": 32,
                "num_epochs": 30,
                "optimiser_args": {
                    "lr": 1e-4,
                },
                "metrics": "r",
                "is_ood": 1,
                # split
                "train_ratio":0.7,
                "valid_ratio":0.1,
                "id_train_val_split":[0.7, 0.1, 0.2],
                # LM
                "use_manifold": 0, 
                "eps": 1, 
                # dti
                "read_dataset": 1, # input dataset directly
                "sub_sample_batch_max_num": 100,
                "store_log": 0,
                "task":"domain_generalization",
                "algorithm":"ERM",
                "hparams": "",
                "hparams_seed": 0,
                "trial_seed": 0,
                "test_envs": [6, 7, 8],
                "output_dir": "train_output",
                "holdout_fraction": 0.2,
                "uda_holdout_fraction": 0., # For domain adaptation, % of test to use unlabeled for training
            },
        }
    return config[dataset]

def get_mixup_config(dataset, manimixup):
    config_dict = dataset_defaults[dataset].copy()
    config_dict["mixtype"] = "random"
    config_dict["use_manifold"] = manimixup
    return config_dict
    args = get_general()
    args_dict = vars(args)
    data_args = {}

    if dataset.upper() == "rcf_mnist".upper():
        data_args = {
            "data_dir" : "dataset/RCF_MNIST",
            "batch_size": 64,
            "num_epochs": 30,
            "optimiser_args": {
                "lr": 0.001,
            },
            "metrics": "rmse",
            "is_ood": 1,
            # split
            "train_ratio":0.7,
            "valid_ratio":0.1,
            "id_train_val_split":[0.7, 0.1, 0.2],
            # mixup
            "mix_alpha": 2.0,
            # rcf
            "use_rotate_class": 0, # 0 -> generate degree randomly; 1 -> sample from 60 fix degree levels
            "spurious_ratio": 0.8,
            "construct_color_data": -1, # 1 -> construct new r-mnist with color spurious information; 0 -> do nothing; -1 -> read rc-fmnist
            "construct_no_color_data": 0, # 1 -> construct r-fmnist; 0 -> do nothing; -1 -> read r-fmnist
            "vis_rcf": 0, # visualize generated data
            "all_pos_color": 0, # 1 -> test data has inverse spurious feature; 0 -> test data has spurious feature as train data
            # ADA
            "use_manifold": 0,
            "eps": 1
        }

    if dataset.upper() == "exchange_rate".upper():
        data_args = { # ref -> LSTNet
            "data_dir": "dataset/exchange_rate/exchange_rate.txt",#"/data/nora/anchor/exchange_rate/exchange_rate.txt", 
            "batch_size": 128,
            "num_epochs": 200,
            "optimiser_args": {
                "lr": 5e-4,
            },
            "metrics": "rmse",
            "is_ood": 0,
            # split
            "train_ratio":0.7,
            "valid_ratio":0.1,
            "id_train_val_split":[0.7, 0.1, 0.2],
            # ts
            "hidCNN": 50, # number of CNN hidden units
            "hidRNN": 50, # number of RNN hidden units
            "window": 24*7, # window size
            "CNN_kernel": 6, # the kernel size of the CNN layers
            "highway_window": 24, # The window size of the highway component
            "clip": 10., # gradient clipping
            "dropout": 0.2, # dropout applied to layers (0 = no dropout)
            "horizon": 12, 
            "skip": 24,
            "hidSkip": 5,
            "L1Loss":False,
            "normalize": 2,
            "output_fun":None,
            # LocalMixup
            "use_manifold": 0,
            "eps": 2.5,

        }
    
    elif dataset.upper() == "electricity".upper():
        data_args = { # ref -> LSTNet
            "data_dir": "dataset/electricity/electricity.txt", #"/data/nora/anchor/electricity/electricity.txt",#
            "batch_size": 128,
            "num_epochs": 200,
            "optimiser_args": {
                "lr": 0.001,#5e-3,
            },
            "metrics": "rmse",
            "is_ood": 0,
            # split
            "train_ratio":0.7,
            "valid_ratio":0.1,
            "id_train_val_split":[0.7, 0.1, 0.2],
            # ts
            "hidCNN": 50, # number of CNN hidden units
            "hidRNN": 50, # number of RNN hidden units
            "window": 24*7, # window size
            "CNN_kernel": 6, # the kernel size of the CNN layers
            "highway_window": 24, # The window size of the highway component
            "clip": 10., # gradient clipping
            "dropout": 0.2, # dropout applied to layers (0 = no dropout)
            "horizon": 24, 
            "skip": 24,
            "hidSkip": 5,
            "L1Loss":False,
            "normalize": 2,
            "output_fun":"Linear",
            # LocalMixup
            "use_manifold": 0,
            "eps": 2.5,
        }
    
    elif dataset.upper() == "dti".upper():
        data_args = {
            "dataset": "Dti_dg",
            "data_dir" : "dataset/dti",
            "batch_size": 32,
            "num_epochs": 20,
            "optimiser_args": {
                "lr": 1e-4,
            },
            "metrics": "r",
            "is_ood": 1,
            # split
            "train_ratio":0.7,
            "valid_ratio":0.1,
            "id_train_val_split":[0.7, 0.1, 0.2],
            # mixup
            "mix_alpha": 2.0,
            "mixtype" : "kde", 
            "kde_bandwidth": 21,
            "use_manifold": 1, 
             # ADA
            "use_manifold": 1,
            "anchor_levels": 24,
            "alpha": 3,
            "cluster": "kmeans",
            # dti
            "read_dataset": 1, # input dataset directly
            "sub_sample_batch_max_num": 100,
            "store_log": 0,
            "task":"domain_generalization",
            "algorithm":"ERM",
            "hparams": "",
            "hparams_seed": 0,
            "trial_seed": 0,
            "test_envs": [6, 7, 8],
            "output_dir": "train_output",
            "holdout_fraction": 0.2,
            "uda_holdout_fraction": 0., # For domain adaptation, % of test to use unlabeled for training
            }

    elif dataset.upper() == "airfoil".upper():
        data_args = {
            "data_dir": "./dataset/UCI/", 
            "input_dim": 5, 
            "batch_size": 16,
            "num_epochs": 150,
            "optimiser_args": {
                "lr": 0.01,
            },
            "metrics": "rmse",
            "is_ood": 0,
            # split
            "train_ratio":0.7,
            "valid_ratio":0.1,
            "id_train_val_split":[0.7, 0.1, 0.2],
            # mixup
            "mix_alpha": 2.0,
            #LM
            "use_manifold": 0,
            "eps": 3.5,
        }
    
    elif dataset.upper() == "no2".upper():
        data_args = {
            "data_dir": "./dataset/NO2/", 
            "input_dim": 7, 
            "batch_size": 32,
            "num_epochs": 120,
            "optimiser_args": {
                "lr": 0.01,
            },
            "metrics": "rmse",
            "is_ood": 0,
            # split
            "train_ratio":0.7,
            "valid_ratio":0.1,
            "id_train_val_split":[0.7, 0.1, 0.2],
            # mixup
            "mix_alpha": 2.0,
            #ADA
            "use_manifold": 0,
            "eps": 2.5,
        }
    
    elif dataset.upper() == "crimes".upper():
        data_args = {
            "data_dir": "./dataset/CommunitiesAndCrime/", 
            "input_dim": 122,
            "batch_size": 48,
            "num_epochs": 250, #200
            "optimiser_args": {
                "lr": 1e-4, #0.001
            },
            "metrics": "rmse",
            "is_ood": 1,
            # split
            "train_ratio":0.7,
            "valid_ratio":0.1,
            "id_train_val_split":[0.7, 0.1, 0.2],
            # mixup
            "mix_alpha": 2.0,
            #ADA
            "use_manifold": 0,
            "eps": 1
        }
    
    elif dataset.upper() == "skillcraft".upper():
        data_args = {
            "data_dir": "./dataset/SkillCraft/", 
            "input_dim": 17,
            "batch_size": 48,
            "num_epochs": 150,
            "optimiser_args": {
                "lr": 0.001,
            },
            "metrics": "rmse",
            "is_ood": 1,
            # split
            "train_ratio":0.7,
            "valid_ratio":0.1,
            "id_train_val_split":[0.7, 0.1, 0.2],
            # mixup
            "mix_alpha": 2.0,
            #LM
            "use_manifold": 0,
            "eps": 1
        }
    
    
    data_args["dataset"] = dataset
    args_dict.update(data_args)
    return args