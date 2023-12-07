import numpy as np
from typing import Tuple
from baselines.CMixup.src.algorithm import get_mixup_sample_rate

def noise_augmentor(X: np.ndarray, y: np.ndarray,
                    k: int =1, adjust_X: bool = False, return_original: bool=True,
                    std: float=0.1, random_state: int=123, std_x: float=0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Vanilla Augmentation by adding Gaussian Noise to the original data. This function will return an augmented dataset  

    Args:
        X (np.ndarray): input variable.
        y (np.ndarray): target variable.
        k (int, optional): How many augmentations should be obtained per original datapoint. Defaults to 1.
        adjust_X (bool, optional): Whether to also add noise to the input variables. Defaults to False.
        return_original (bool, optional): Whether to also return the original data. Defaults to True.
        std (float, optional): Standard deviation of Gaussian noise added to the data. Defaults to 0.1.
        random_state (int, optional): RandomState. Defaults to 123.
        std_x (float, optional): If adjust_X is true, this determine the standard deviation of Gaussian noise added to X. Defaults to 0.05.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Augmented X, Augmented Y
    """

    state = np.random.RandomState(random_state)
    n = X.shape[0]
    p = X.shape[1]
    if len(y.shape) == 1: y = y.reshape(-1, 1)

    if return_original:
        rows = n*k + n
        X_til = np.zeros((rows, p))
        y_til = np.zeros((rows, 1))
        X_til[(n*k):,] = X
        y_til[(n*k):,] = y
    else:
        rows = n*k
        X_til = np.zeros((rows, p))
        y_til = np.zeros((rows, 1))

    for i in range(k):
        y_til[i * n:(i + 1) * n, :] = y + state.normal(0, std, n).reshape(-1, 1)

        if adjust_X:
            X_til[i * n:(i + 1) * n, :] = X + state.multivariate_normal(np.zeros(p),std_x * np.identity(p), n)
        else:
            X_til[i * n:(i + 1) * n, :] = X

    return X_til, y_til


def cmixup_augmentor(X, y, args, k: int =1, return_original: bool=True)  -> Tuple[np.ndarray, np.ndarray]:
    """_summary_

    Args:
        X (_type_): input variable.
        y (_type_): target variable.
        args (_type_): args for creating CMixup sample rate (according to C-Mixup repository implementation). Needs to have "mixtype", "show_process", "kde_type", "kde_bandwidth"
        k (int, optional): How many augmentations should be obtained per original datapoint.. Defaults to 1.
        return_original (bool, optional): Whether to also return the original data. Defaults to True.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Augmented X, Augmented Y
    """
    data_packet = {
        "x_train": X, 
        "y_train": y, 
    }

    mixup_idx_sample_rate = get_mixup_sample_rate(args, data_packet, "cpu")

    n = X.shape[0]
    p = X.shape[1]
    if len(y.shape) == 1: y = y.reshape(-1, 1)

    if return_original:
        rows = n*k + n
        X_til = np.zeros((rows, p))
        y_til = np.zeros((rows, 1))
        X_til[(n*k):,] = X
        y_til[(n*k):,] = y
    else:
        rows = n*k
        X_til = np.zeros((rows, p))
        y_til = np.zeros((rows, 1))

    for i in range(k):
        lambd = np.random.beta(args.mix_alpha, args.mix_alpha)

        shuffle_idx = np.arange(X.shape[0])
        idx_1 = shuffle_idx
        idx_2 = np.array([np.random.choice(np.arange(X.shape[0]), p=mixup_idx_sample_rate[sel_idx]) for sel_idx in idx_1])
        X1 = X[idx_1]
        Y1 = y[idx_1]
        X2 = X[idx_2]
        Y2 = y[idx_2]

        mixup_Y = Y1 * lambd + Y2 * (1 - lambd)
        mixup_X = X1 * lambd + X2 * (1 - lambd)

        y_til[i * n:(i + 1) * n, :] = mixup_Y
        X_til[i * n:(i + 1) * n, :] = mixup_X
 
    return X_til, y_til
