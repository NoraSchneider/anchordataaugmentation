import numpy as np
from typing import List, Tuple, Optional
from enum import Enum
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from ..models.MLPs import MLP
from ..utils.train import train, augmentationtype
from ..utils.val import test


class ModelDescription(Enum):
    mlp = "MLP"
    reg = "RidgeRegression"

class LinearData:
    """Generating linear regression data and train different models on it.
    Args:
        dim (int, optional): _description_. Defaults to 19.
        random_state_train (int, optional): _description_. Defaults to 456.
        random_state_test (int, optional): _description_. Defaults to 314.
        size (int, optional): _description_. Defaults to 640.
    """

    def __init__(self, dim: int = 19, random_state_train: int=456, random_state_test: int=314, size=640):
        state = np.random.RandomState(random_state_test)
        self.w = state.normal(0, 1, dim)
        self.b = state.normal(0, 1, 1)
        self.X_train, self.y_train = self.generate_dataset(size=size, random_state=random_state_train)
        self.X_test, self.y_test = self.generate_dataset(size=100000, random_state=random_state_test)

    def generate_dataset(self, size: int, random_state: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a dataset according to the parameters of this class

        Args:
            size (int): Number of samples
            random_state (int): defines random state

        Returns:
            Tuple[np.ndarray, np.ndarray]: Covariates array X, target vector y
        """
        p = self.w.shape[0]
        state = np.random.RandomState(random_state)
        X = state.standard_normal(size=(size, p))
        y = np.matmul(X, self.w) + self.b + state.normal(0, 0.1, size)
        return X, y

    def linear_fit(self, X_train: Optional[np.ndarray] = None, y_train: Optional[np.ndarray] = None) -> List[float]:
        """Fit a ridge regression model

        Args:
            X_train (Optional[np.ndarray], optional): Training covariates (if None, then the class data X_train is used). Defaults to None.
            y_train (Optional[np.ndarray], optional): Training target variables (if None, then the class data y_train is used). Defaults to None.

        Raises:
            ValueError: 

        Returns:
            List[float]: mse, model
        """
        if (X_train is None and y_train is not None) or (y_train is None and X_train is not None):
            raise ValueError("both X_train and y_train have to be passed.")

        if X_train is not None:
            X = X_train
            y = y_train
        else:
            X = self.X_train
            y = self.y_train

        model = Ridge().fit(X, y)
        preds = model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, preds)

        return mse, model

    def mlp_fit_torch(self, args, mode, hidden_units: int, X_train: Optional[np.ndarray] = None, y_train: Optional[np.ndarray] = None) -> float:
        """_summary_

        Args:
            args (_type_): args for training the model.
            mode (_type_): Which augmentation mode to train the model (e.g. ada, erm, cmixup, vanilla).
            hidden_units (int): Number of hidden units.
            X_train (Optional[np.ndarray], optional): X_train. Defaults to None.
            y_train (Optional[np.ndarray], optional): y_train. Defaults to None.

        Returns:
            float: MSE
        """
        model = MLP(input_units=19, hidden_units=hidden_units)
        datapacket = {
            "x_train": X_train, 
            "y_train": y_train
        }

        if hasattr(args, "anchors"):
            datapacket["anchors"] = args.anchors

        best_model, best_mse = train(args, model, datapacket, ts_data=None, verbose=False, is_anchor=(mode==augmentationtype.anchor), is_mixup=(mode==augmentationtype.cmixup), is_vanilla=(mode==augmentationtype.vanilla), device="cuda" if args.cuda else "cpu")
            
        result_dict_best = test(best_model, self.X_test, self.y_test, device="cuda" if args.cuda else "cpu")

        return result_dict_best["mse"]


    def fit(self, model: ModelDescription, X_train: Optional[np.ndarray] = None, y_train: Optional[np.ndarray] = None,  hidden_units: Optional[int]=None, return_model=False, args=None, mode=None) -> float:
        """Fit a model on the data according to the specifications

        Args:
            model (ModelDescription): which model is fitted (Ridge regression or a MLP).
            X_train (Optional[np.ndarray], optional): X_train. Defaults to None.
            y_train (Optional[np.ndarray], optional): y_train. Defaults to None.
            hidden_units (Optional[int], optional): Number of hidden units. Defaults to None.
            return_model (bool, optional): If to return the model. Defaults to False.
            args (_type_, optional): args used for model training. Defaults to None.
            mode (_type_, optional): Which augmentation mode to train the model (e.g. ada, erm, cmixup, vanilla). Defaults to None.

        Returns:
            float: MSE
        """
        
        if model.value == ModelDescription.mlp.value:
            assert hidden_units is not None and args is not None
            return self.mlp_fit_torch(args, mode, hidden_units, X_train, y_train)

        elif model.value == ModelDescription.reg.value:
            if return_model:
                return self.linear_fit(X_train, y_train)
            else:
                return self.linear_fit(X_train, y_train)[0]