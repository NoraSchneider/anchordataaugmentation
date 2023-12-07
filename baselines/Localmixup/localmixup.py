# code base: https://github.com/raphael-baena/Local-Mixup/tree/main (Baena et. al., 2022) 
import numpy as np
import torch
import torch.nn as nn
import numpy as np

def mixup_data(x, y, eps = 0, alpha=1.0, use_cuda=False):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index,:]

    #we compute the weights here. If d> eps => w = 0.
    m = nn.Threshold(eps,0,inplace=False)
    Cdist = torch.norm(x.flatten(start_dim = 1)-x[index,:].flatten(start_dim = 1),p=2,dim =1) #Euclidean distance between x and x permuted
    dist = torch.heaviside(-m(Cdist),torch.tensor(1.)) #we apply the threshold. 
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam,dist
 
def mix_criterion(criterion, pred, y_a, y_b,lam):

    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)   




