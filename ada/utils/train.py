import copy
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from ada import ADA, clustered_anchor
from tqdm import tqdm
from enum import Enum
from typing import Dict
from baselines.CMixup.src.algorithm import get_batch_kde_mixup_batch
from baselines.Augmentors import * 
from baselines.Localmixup.localmixup import mix_criterion, mixup_data
from .val import test

# This is a helper class for choosing which augmentation methods can be used when training a Pytorch model. 
class augmentationtype(Enum): 
    erm = "ERM"
    cmixup = "CMixup"
    anchor = "Anchor"
    vanilla = "Vanilla"
    localmixup = "Localmixup"

def train(args, 
          model, 
          datapacket: Dict, 
          ts_data: None, 
          is_mixup: bool=False, 
          is_anchor: bool = False, 
          is_vanilla: bool = False, 
          is_localmixup: bool=False,
          device: str="cpu", 
          verbose: bool = True, 
          vanillaaugment_x: bool = False,
          early_stopping=False,
          n_iter_no_improvement=200,
          tol = 1e-4,
          ):
    """Train a Pytorch model with a specified augmentation method or without augmentation (=ERM).

    Args:
        args (_type_): _description_
        model (_type_): Pytorch model.
        datapacket (_type_): Dictionary which data with keys "x_train", "y_train" and optional also "x_valid" and "y_valid".
        ts_data (None): Timeseries data.
        is_mixup (bool, optional): True for Manimixup, mixup and cmixup. The exact augmentation type is defined via the args file (e.g. mixtype=kde is equivalent to cmixup, use_manifold=1 is equivalent to manimixup). Defaults to False.
        is_anchor (bool, optional): True for ADA. Defaults to False.
        is_vanilla (bool, optional): True for vanilla augmentation. Defaults to False.
        is_localmixup (bool, optional): True for local mixup. Defaults to False.
        device (str, optional): Which device to train the model on. Defaults to "cpu".
        verbose (bool, optional): True for extra print statements. Defaults to True.
        vanillaaugment_x (bool, optional): For vanilalaugmentation if to also augment the covariates. Defaults to False.
        early_stopping (bool, optional): True for using early stopping. Defaults to False.
        n_iter_no_improvement (int, optional): _description_. Defaults to 10000.
        tol (_type_, optional): For early stopping. Defaults to 1e-4.

    Returns:
        _type_: trained model, mse of best model
    """

    model.train(True)
    model.to(device)
    optimizer = Adam(model.parameters(), **args.optimiser_args)
    loss_fun = nn.MSELoss(reduction="mean")

    best_mse = 1e10  # for best update
    iters_no_improvement = 0

    x_train = datapacket['x_train']
    y_train = datapacket['y_train']

    if "x_valid" in datapacket.keys(): # if we have a validation dataset, we use this to determine when to stop
        x_valid = datapacket['x_valid']
        y_valid = datapacket['y_valid']
    else: # if not we evaluate our model on training data
        x_valid = datapacket['x_train']
        y_valid = datapacket['y_train']
    
    #this might cut off some samples at the end, but is exactly equivalent to C-Mixup implementation and for comparability we replicated this
    if args.dataset in ["Airfoil", "NO2", "TimeSeries", "RCF_MNIST", "CommunitiesAndCrime", "SkillCraft", "Dti_dg"]:
        iteration = len(x_train) // args.batch_size 
    else:
        iteration = int(np.ceil(len(x_train)/args.batch_size))
    
    # Set parameters for different augmentation methods
    if is_anchor and "anchors" in datapacket.keys():
        predefined_anchors = True
        anchors_x_train = datapacket["anchors"]["x_train"]
        anchors_x_train = torch.tensor(anchors_x_train, dtype=torch.float32)
    else:
        predefined_anchors=False
        anchors_x_train = None

    if is_mixup: mixup_idx_sample_rate = args.mixup_idx_sample_rate
    if is_anchor: gammas = args.gammas
    if is_vanilla: std = args.std

    need_shuffle = not args.is_ood

    # Train the model
    for epoch in tqdm(range(args.num_epochs)):
        torch.cuda.empty_cache()
        if early_stopping and iters_no_improvement == n_iter_no_improvement:
            return model, best_mse

        model.train()
        shuffle_idx = np.random.permutation(np.arange(len(x_train)))
        if need_shuffle: # id
            x_train_input = x_train[shuffle_idx]
            y_train_input = y_train[shuffle_idx]
            if predefined_anchors:
                anchors_x_train_input = anchors_x_train[shuffle_idx]
        else: #ood
            x_train_input = x_train
            y_train_input = y_train
            if predefined_anchors:
                anchors_x_train_input = anchors_x_train

        for idx in range(iteration):
            # select batch
            x_input_tmp = x_train_input[idx * args.batch_size:(idx + 1) * args.batch_size]
            y_input_tmp = y_train_input[idx * args.batch_size:(idx + 1) * args.batch_size]

            if is_anchor:
                loss = _anchor_iteration(x_input_tmp=x_input_tmp, y_input_tmp=y_input_tmp, idx=idx, args=args, model=model,loss_fun=loss_fun, anchors_x_train_input=anchors_x_train_input, gammas=gammas, device=device, ts_data=ts_data)
            elif is_mixup:
                lambd = np.random.beta(args.mix_alpha, args.mix_alpha)
                if need_shuffle: # get batch idx
                    idx_1 = shuffle_idx[idx * args.batch_size:(idx + 1) * args.batch_size]
                else:
                    idx_1 = np.arange(len(x_train))[idx * args.batch_size:(idx + 1) * args.batch_size]
                if args.mixtype == 'kde': 
                    idx_2 = np.array(
                        [np.random.choice(np.arange(x_train.shape[0]), p=mixup_idx_sample_rate[sel_idx]) for sel_idx in
                        idx_1])
                else: # random mix
                    idx_2 = np.array(
                        [np.random.choice(np.arange(x_train.shape[0])) for sel_idx in idx_1])
                X1 = torch.tensor(x_train[idx_1], dtype=torch.float32)
                Y1 = torch.tensor(y_train[idx_1], dtype=torch.float32)
                X2 = torch.tensor(x_train[idx_2], dtype=torch.float32)
                Y2 = torch.tensor(y_train[idx_2], dtype=torch.float32)
                loss = _mixup_iteration(X1=X1, Y1=Y1, X2=X2, Y2=Y2, lambd=lambd, args=args, model=model, loss_fun=loss_fun, device=device, ts_data=ts_data)
            
            elif is_vanilla:
                loss = _vanilla_iteration(x_input_tmp, y_input_tmp,args, model, loss_fun, device, std, vanillaaugment_x, ts_data)
            
            elif is_localmixup:
                x_input = torch.tensor(x_input_tmp, dtype=torch.float32).to(device)
                y_input = torch.tensor(y_input_tmp, dtype=torch.float32).to(device)
                loss = _localmixup_iteration(x_input=x_input, y_input=y_input, args=args, model=model, loss_fun=loss_fun, eps=args.eps, device=device)
            
            else: #ERM 
                x_input = torch.tensor(x_input_tmp, dtype=torch.float32).to(device)
                y_input = torch.tensor(y_input_tmp, dtype=torch.float32).to(device)
                pred_Y = model(x_input)
                loss = loss_fun(pred_Y, y_input) 
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss.detach()

        # validation
        result_dict = test(model, x_valid, y_valid, device, batchsize=16)
        if verbose:
            print(f"Epoch {epoch}: {result_dict}", flush=True)

        if result_dict["mse"] <= best_mse:
            best_mse = result_dict["mse"]
            best_mse_model = copy.deepcopy(model)
            if verbose:
                print(f'update best mse! epoch = {epoch}') 
            iters_no_improvement = 0
        
        elif early_stopping:
            iters_no_improvement += 1 
        
    return best_mse_model, best_mse

#### Which augmentation style to be used (if not ERM)

def _anchor_iteration(x_input_tmp, y_input_tmp, idx, args, model, loss_fun, anchors_x_train_input, gammas, device, ts_data):
    # sample gamma
    gamma = gammas[np.random.randint(0, len(gammas))]

    # use global anchors
    if anchors_x_train_input is not None:
        anchors_x_train_tmp = anchors_x_train_input[idx * args.batch_size:(idx + 1) * args.batch_size]
        idx = torch.nonzero(torch.all(anchors_x_train_tmp[..., :] == 0, dim=0))
        anchors_x_train_tmp = torch.tensor(np.delete(anchors_x_train_tmp.cpu().numpy(), idx.cpu().numpy(), axis=1))
    
    # generate anchor for batch (TODO: Adapt)
    else:
        if x_input_tmp.shape[0] >= args.anchor_levels:
            k = args.anchor_levels
        else:
            k = x_input_tmp.shape[0]
        if len(x_input_tmp.shape) == 3:
            anchors_x_train_tmp = clustered_anchor(x_input_tmp, y_input_tmp, anchor_levels=k)
        else:
            anchors_x_train_tmp = clustered_anchor(y=y_input_tmp, anchor_levels=k)
    
    #print(x_input_tmp.get_device(), y_input_tmp.get_device(), anchors[0].get_device())
    if isinstance(x_input_tmp,np.ndarray):
        x_input_tmp = torch.tensor(x_input_tmp, dtype=torch.float32)
        y_input_tmp = torch.tensor(y_input_tmp, dtype=torch.float32) 


    anchor_X, anchor_y = ADA.transform_pytorch(X=x_input_tmp, y=y_input_tmp, gamma=gamma, anchorMatrix=anchors_x_train_tmp)
    
    anchor_X = anchor_X.to(device)
    anchor_y = anchor_y.to(device)
    x_input = x_input_tmp.to(device)
    anchor_matrix_pytorch = anchors_x_train_tmp.to(device)

    if args.use_manifold:
        model.to(device)
        pred_Y = model.forward_anchor(x_input.to(device), gamma,anchor_matrix_pytorch.to(device))
    else:
        pred_Y = model.forward(anchor_X)

    if args.dataset in ["exchange_rate".upper(), "electricity".upper()]: # time series loss need scale
        scale = ts_data.scale.expand(pred_Y.size(0),ts_data.m)
        loss = loss_fun(pred_Y * scale, anchor_y * scale)
    else:    
        loss = loss_fun(pred_Y.squeeze().float(), anchor_y.squeeze().float())
    
    return loss

def _vanilla_iteration(x_input_tmp, y_input_tmp, args, model, loss_fun, device, std, ts_data, vanillaaugment_x=False):
    vanilla_y = y_input_tmp + np.random.normal(0, std, y_input_tmp.shape[0]).reshape(-1, 1)

    if vanillaaugment_x:
        x_input_tmp = x_input_tmp + np.random.normal(0,  args.std_x, x_input_tmp.shape[0]).reshape(-1, 1) #todo 
                    
    # -> tensor
    x_input = torch.tensor(x_input_tmp, dtype=torch.float32).to(device)
    vanilla_y = torch.tensor(vanilla_y, dtype=torch.float32).to(device)

    # forward
    pred_Y = model(x_input)

    if args.dataset in ["exchange_rate".upper(), "electricity".upper()]: # time series loss need scale
        scale = ts_data.scale.expand(pred_Y.size(0),ts_data.m)
        loss = loss_fun(pred_Y * scale, vanilla_y * scale)
    else:    
        loss = loss_fun(pred_Y, vanilla_y)

    return loss

def _mixup_iteration(X1, Y1, X2, Y2, lambd, args, model, loss_fun, device, ts_data):
    if args.batch_type == 1: # sample from batch
        assert args.mixtype == 'random'
        if not repr_flag: # show the sample status once
            args.show_process = 0
        else:
            repr_flag = 0
        X2, Y2 = get_batch_kde_mixup_batch(args,X1,X2,Y1,Y2,device)
        args.show_process = 1

    X1 = X1.to(device)
    X2 = X2.to(device)
    Y1 = Y1.to(device)
    Y2 = Y2.to(device)

    # mixup
    mixup_Y = Y1 * lambd + Y2 * (1 - lambd)
    mixup_X = X1 * lambd + X2 * (1 - lambd)
    # forward
    if args.use_manifold == True:
        pred_Y = model.forward_mixup(X1, X2, lambd)
    else:
        pred_Y = model.forward(mixup_X)

    if args.dataset in ["exchange_rate".upper(), "electricity".upper()]: # time series loss need scale
        scale = ts_data.scale.expand(pred_Y.size(0),ts_data.m)
        loss = loss_fun(pred_Y * scale, mixup_Y * scale)
    else:    
        loss = loss_fun(pred_Y.squeeze(), mixup_Y.squeeze())
    return loss

def _localmixup_iteration(x_input, y_input, args, model, loss_fun, eps, device):
    if eps==0:
        is_mixup=False
    else:
        is_mixup=True

    if is_mixup:
        if args.use_manifold:
            out, y_a, y_b, lam, dist = model.forward_localmixup(x_input, y_input, eps=eps)
        else:
            x_input, y_a, y_b, lam,dist = mixup_data(x_input,y_input,eps=eps)
            out = model(x_input)
        loss = mix_criterion(loss_fun, out.flatten().float(),y_a.flatten().float(),y_b.flatten().float(),lam)
    else:
        out = model(x_input)
        y_a, y_b, lam = None, None, None
        dist = torch.ones(x_input.size()[0])
        loss = loss_fun(out, y_input.flatten().float())
    
    loss = (loss * dist).mean()

    return loss