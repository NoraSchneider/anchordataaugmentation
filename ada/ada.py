import numpy as np
import pandas as pd
import torch
import math
from typing import List, Optional, Dict, Union
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder

class ADA:
    """This class implements ADA: Anchor Data Augmentation (Schneider et al., 2023)

    The code is implemented for either numpy arrays or pytorch tensors.
    Args:
            X (Union[np.ndarray, pd.DataFrame, torch.tensor]): Feature matrix of shape (n x, ....)
            y (Union[np.ndarray, pd.DataFrame, torch.tensor]): Target matrix of shape (n x, ...)
            anchor (_type_, optional): Anchor matrix. Defaults to None.
            generate_anchor_args (Optional[Dict], optional): If anchor matrix is not specified, this class
                                                            generates an Anchor Matrix as the onehot encoding
                                                            of a KMeans clustering of the data. To do so a dictionary
                                                            has to have the keys "type" and "anchor_levels". Defaults to None.
        Raises:
            Exception: raise exception if X or y are neither numpy.ndarray, pandas.DataFrame or torch.is_tensor. 
    """

    def __init__(self, X:Union[np.ndarray, pd.DataFrame, torch.tensor], y:Union[np.ndarray, pd.DataFrame, torch.tensor], anchor = None, generate_anchor_args: Optional[Dict] = None):
        self.X = X
        self.y = y
        assert anchor is not None or generate_anchor_args is not None
        # create Anchor matrix/ matrices A
        if anchor is not None: 
            assert anchor.shape[0] == X.shape[0]
        else:
            assert set(["anchor_levels"]).issubset(generate_anchor_args.keys())
            # kmeans clustering for data
            anchor = clustered_anchor(X, y, generate_anchor_args["anchor_levels"])
        self.anchor = anchor
        if torch.is_tensor(self.X) and (torch.is_tensor(self.y) or self.y is None):
            self._transform = ADA.transform_pytorch
        elif isinstance(self.X, np.ndarray) and (isinstance(self.y, np.ndarray)):
            if len(self.y.shape) == 1: self.y = self.y.reshape(-1, 1)
            self._transform = ADA.transform_numpy
        elif isinstance(X, pd.DataFrame) and (isinstance(y, pd.DataFrame)):
            self.X = self.X.to_numpy()
            self.y = self.y.to_numpy()
            if len(self.y.shape) == 1: self.y = self.y.reshape(-1, 1)
            self._transform = ADA.transform_numpy
        else: 
            raise Exception("X and y have to be of the same datatype (tensor, numpy array, pandas).")

    def augment(self, 
                gamma: Optional[Union[int, List]]=None, 
                generate_gamma_args: Optional[Dict] = None, 
                return_original_data:bool=False,
                return_list:bool=False, 
                random_state: int=123):
        """augments X & y according to ADA strategy for a given value of gamma or a list of gamma values. 

        Args:
            gamma (Optional[Union[int, List]], optional): value for gamma or list of gamma values. Defaults to None.
            generate_gamma_args (Optional[Dict], optional): If gamma is None, then it is uniformly sampled. Defaults to None.
            return_original_data (bool, optional): Whether to return the original data. Defaults to False.
            random_state (int, optional): random state. Defaults to 123.

        Returns:
            List: returns a list of tuples [(X_{til, \gamma_1}, y_{til, \gamma_1}), ... ] 
        """
        
        state = np.random.RandomState(random_state)
        # determine gammas
        if generate_gamma_args is not None:
            assert set(["n_gamma", "gamma"]).issubset(generate_gamma_args.keys())
            n_gamma = generate_gamma_args["n_gamma"]
            gamma = list(state.uniform(generate_gamma_args["gamma"][0], generate_gamma_args["gamma"][1], n_gamma))
        else:
            if isinstance(gamma, int): 
                gamma = [gamma]
            n_gamma = len(gamma)

        augmented_samples = []
        for g in gamma:
            X_til, y_til = self._transform(X=self.X, y=self.y, gamma=g, anchorMatrix=self.anchor)
            augmented_samples.append((X_til, y_til))
        
        if return_original_data:
            augmented_samples.append((self.X, self.y))
        
        if return_list:
            return augmented_samples
        else: 
            return np.concatenate([x for x, y in augmented_samples]), np.concatenate([y for x,y in augmented_samples])

    
    def transform_numpy(*, X: np.ndarray, y: np.ndarray, gamma=2, anchorMatrix: np.ndarray):
        """transforms X and y (numpy arrays) according to ADA strategy for a given value of gamma and anchor matrix.
        Args:
            X (np.ndarray): feature array of shape (n x ...).
            y (np.ndarray, optional): target array of shape (n x ...). 
            gamma (int, optional): value for gamma used for ADA. Defaults to 2.
            anchorMatrix (np.ndarray): anchor matrix A of shape (n x p).
        Returns:
            Tuple[np.ndarray]: (X_til, y_til)
        """
        if len(y.shape) == 1: y = y.reshape(-1, 1)
        if len(X.shape) >= 3:
            # multidimensional input is being flattened
            number_dims_X = len(X.shape) - 1
            X_flat = X.reshape((X.shape[0], math.prod([i for i in X.shape[1:]])))
            X_til, y_til = ADA.transform_numpy(X=X_flat, y=y, gamma=gamma, anchorMatrix=anchorMatrix)
            X_til  = X_til.reshape(X.shape)
        else:
            n = X.shape[0]
            y_mean = np.mean(y)
            X_mean = np.mean(X, axis=0)

            # Projection matrix
            p = np.matmul(np.matmul(anchorMatrix, np.linalg.inv(np.matmul(anchorMatrix.T, anchorMatrix))), anchorMatrix.T)
            p_row_sum = np.sum(p, axis=1)
            scaler = 1 + (np.sqrt(gamma) - 1) * p_row_sum
            X_til = np.matmul(np.identity(n) + (np.sqrt(gamma) - 1) * p, X - X_mean)/scaler[:,None] + X_mean
            y_til = np.matmul(np.identity(n) + (np.sqrt(gamma) - 1) * p, (y - y_mean))/scaler[:, None] + y_mean
        assert X_til.shape == X.shape
        assert y_til.shape == y.shape
        return X_til, y_til

    def transform_pytorch(*, X: torch.tensor, y: Optional[torch.tensor]=None, gamma=2, anchorMatrix: torch.tensor):
        # TODO y can be none
        """transforms X and y (torch tensors) according to ADA strategy for a given value of gamma and anchor matrix.

        Args:
            X (torch.tensor): feature tensor of shape (n x ...).
            y (torch.tensor): target array of shape (n x ...).
            gamma (int, optional): value for gamma used for ADA. Defaults to 2.
            anchorMatrix (torch.tensor): anchor matrix A of shape (n x p).

        Returns:
            Tuple[torch.tensor]: (X_til, y_til)
        """

        if y is not None:
            assert X.get_device() == y.get_device()
            if len(y.shape) == 1: y = y.view(-1, 1)
        
        if anchorMatrix is not None and not torch.is_tensor(anchorMatrix):
            anchorMatrix = torch.from_numpy(anchorMatrix)

        device = torch.device('cuda' if X.get_device()==0 else 'cpu')
    
        n = X.shape[0]

        if len(X.shape) >= 3:
            # multidimensional input is being flattened
            number_dims_X = len(X.shape) - 1
            X_flat = X.flatten(start_dim=1, end_dim=number_dims_X)
            X_til, y_til= ADA.transform_pytorch(X=X_flat, y=y, gamma=gamma, anchorMatrix=anchorMatrix)
            X_til  = X_til.reshape(X.shape)
        
        else:
            a = anchorMatrix.to(device)
            X_til = torch.zeros(size=X.shape).to(device)
            y_til = None
            p = torch.matmul(torch.matmul(a, torch.inverse(torch.matmul(a.T, a))), a.T).to(device)
            p_row_sum = torch.sum(p, dim=1)
            scaler = 1 + (np.sqrt(gamma) - 1) * p_row_sum
            Id = torch.eye(n).to(device)
            
            if y is not None:
                y_mean = torch.mean(y, dim=0)
                y_til = torch.matmul((Id+ (np.sqrt(gamma) - 1) * p).float(), (y - y_mean).float())/ scaler[:, None] + y_mean
        
            X_mean = torch.mean(X, dim=0)
            X_til = torch.matmul((Id + (np.sqrt(gamma) - 1) * p).float(), (X - X_mean).float()) / scaler[:, None] + X_mean
        
        assert X_til.shape == X.shape
        if y is not None:
            assert y_til.shape == y.shape
        return X_til, y_til
    
def get_gammas(a: int, n: int) -> List[float]:
    """Get a list of possible gamma values to perform ADA. They are 

    Args:
        a (int): Determines the range of the gamma values. 
        n (int): Determines the number of possible values of gamma. 

    Returns:
        List[float]: List with gamma values. 
    """
    steps = int(n / 2)
    b = (a - 1) / steps
    gammas = [np.round(a - i * b, 4) for i in range(steps)]
    gammas += [np.round(1 / g, 4) for g in reversed(gammas)]

    return gammas

def clustered_anchor(X: np.ndarray=None, y:np.ndarray = None, anchor_levels: int=2) -> np.ndarray:
    """Obtain the Anchor matrix for a dataset based on onehot encoding of KMeans clustering. 

    Args:
        X (np.ndarray): _description_
        y (Optional[np.ndarray], optional): _description_. Defaults to None.
        anchor_levels (int, optional): Number of clusters. Defaults to 2.

    Returns:
        np.ndarray: onehot encoded cluster labels.
    """
    assert X is not None or y is not None

    n = X.shape[0]

    if X is not None and len(X.shape) > 2:
        X = X.reshape((n, math.prod([i for i in X.shape[1:]])))
    if y is not None: 
        if len(y.shape)<=1: y = y.reshape(-1, 1)
        data = np.column_stack((X, y))
    else: data=X
    
    if n <= anchor_levels:
        anchor_levels = n

    kmeans = KMeans(anchor_levels).fit(data)
    cluster_labels = kmeans.predict(data)
    onehot_encoder = OneHotEncoder(sparse=False)
    anchor = onehot_encoder.fit_transform(cluster_labels.reshape(-1, 1))
    return anchor