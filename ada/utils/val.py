import torch.nn as nn
import torch
from typing import List
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import numpy as np


def test(model, x_list, y_list, device, batchsize=None):
    """Evaluate a model based on different metrics (mse, rmse, r^2, mape).

    Args:
        model (_type_): Model.
        x_list (_type_): x_test
        y_list (_type_): y_test
        device (_type_): device to use
        batchsize (_type_, optional): batchsize. Defaults to None.

    Returns:
        _type_: dictionary with performance metrics.
    """
    
    model.eval()
    with torch.no_grad():

        if batchsize:
            val_len=batchsize
            val_iter= int(np.ceil(x_list.shape[0]/batchsize))
        
        else:
            val_iter = 1
            val_len = x_list.shape[0]

        y_list_pred = []
        assert val_iter >= 1 #  easy test

        for ith in range(val_iter):

            if isinstance(x_list,np.ndarray):
                x_list_torch = torch.tensor(x_list[ith*val_len:(ith+1)*val_len], dtype=torch.float32).to(device)
            else:
                x_list_torch = x_list[ith*val_len:(ith+1)*val_len].to(device)

            model = model.to(device)
            pred_y = model(x_list_torch).cpu().numpy()
            y_list_pred.append(pred_y)

        y_list_pred = np.concatenate(y_list_pred,axis=0)
        y_list = y_list.squeeze()
        y_list_pred = y_list_pred.squeeze()

        if not isinstance(y_list, np.ndarray):
            y_list = y_list.numpy()
        
        ###### calculate metrics ######

        mean_p = y_list_pred.mean(axis = 0)
        sigma_p = y_list_pred.std(axis = 0)
        mean_g = y_list.mean(axis = 0)
        sigma_g = y_list.std(axis = 0)

        index = (sigma_g!=0)
        corr = ((y_list_pred - mean_p) * (y_list - mean_g)).mean(axis = 0) / (sigma_p * sigma_g)
        corr = (corr[index]).mean()
        #corr = 0
        mse = (np.square(y_list_pred  - y_list )).mean()
        result_dict = {'mse':mse, 'r':corr, 'r^2':corr**2, 'rmse':np.sqrt(mse)}

        not_zero_idx = y_list != 0.0
        mape = (np.fabs(y_list_pred[not_zero_idx] -  y_list[not_zero_idx]) / np.fabs(y_list[not_zero_idx])).mean() * 100
        result_dict['mape'] = mape
    
    return result_dict


# code base: https://github.com/huaxiuyao/C-Mixup (Yao et. al., 2022) 
def cal_worst_acc(args,data_packet,best_model_rmse,best_result_dict_rmse,ts_data,device, phase: str = "test", ):
    #### worst group acc ---> rmse ####
    if args.is_ood:
        x_test_assay_list = data_packet[f'x_{phase}_assay_list']
        y_test_assay_list = data_packet[f'y_{phase}_assay_list']
        worst_acc = 0.0 if args.metrics == 'rmse' else 1e10
            
        for i in range(len(x_test_assay_list)):
            result_dic = test(best_model_rmse,x_test_assay_list[i],y_test_assay_list[i],device, batchsize=64)
            acc = result_dic[args.metrics] 
            if args.metrics == 'rmse':
                if acc > worst_acc:
                    worst_acc = acc
            else:#r
                if np.abs(acc) < np.abs(worst_acc):
                    worst_acc = acc
        print('worst {} = {:.3f}'.format(args.metrics, worst_acc))
        best_result_dict_rmse['worst_' + args.metrics] = worst_acc
    
    return best_result_dict_rmse


# not used
def val(valloader, criterion, model, device):
    """function for validating the model (essentially this is similar to test but works with a dataloader)

    Args:
        valloader (_type_): Validation loader.
        criterion (_type_): Criterion to use.
        model (_type_): Model.
        device (_type_): Device.

    Returns:
        tuple: criterion, mape, r^2 
    """
    val_loss = 0
    mape = 0
    model.to(device)
    model.eval()

    all_ys = []
    all_preds = []

    for j, (X, y) in enumerate(valloader):
        with torch.no_grad():
            preds = model(X.to(device).float())
            loss = criterion(y.to(device).float().squeeze(), preds.squeeze())
        
        all_ys.append(y.numpy())
        all_preds.append(preds.numpy())
        val_loss += loss.item() * X.shape[0]

        mape += mean_absolute_percentage_error(y.cpu().numpy(), preds.cpu().numpy()) * X.shape[0]

    val_loss = val_loss/ len(valloader.dataset)

    all_ys = np.concatenate(all_ys)
    all_preds = np.concatenate(all_preds)

    mape = mean_absolute_percentage_error(all_ys, all_preds)
    r2 = r2_score(all_ys, all_preds)
    return val_loss, mape, r2

