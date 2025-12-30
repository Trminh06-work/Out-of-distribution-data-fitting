import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
import xgboost as xgb
from lightgbm import LGBMRegressor

from pytabkit import RealMLP_TD_Regressor
from ft_transformer import FTTransformer

from sklearn.base import BaseEstimator

import logging

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings(
    "ignore",
    message=r".*tensorboardX.*removed.*",
    category=UserWarning,
    module=r"pytorch_lightning.*",
)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning").setLevel(logging.ERROR)

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from EvaluationToolbox import Evaluator, ModelSketcher, OptunaTuner


@dataclass
class ModelConfig:
    use_optim: bool = True
    metric: str = "rmse"
    n_splits: int = 5
    seed: int = 42



def pick_device(prefer_mps: bool = True, prefer_cuda: bool = True) -> str:
        """Pick the best available accelerator with a slight preference ordering."""
        if prefer_cuda and torch.cuda.is_available():
            return "cuda"
        if prefer_mps and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"



class BaseTabularRegressor(ABC):
    def __init__(
        self,
        df_train: pd.DataFrame, df_test: pd.DataFrame,
        config: ModelConfig,
    ):
        self.df_train = df_train
        self.df_test = df_test
        self.config = config

        # Split X/y (assumes last column is target)
        if df_train is not None:
            self.X_train = df_train.iloc[:, :-1]
            self.y_train = df_train.iloc[:, -1]
        if df_test is not None:
            self.X_test = df_test.iloc[:, :-1]
            self.y_test = df_test.iloc[:, -1]

        self.model: BaseEstimator = self.build_model()

        if self.config.use_optim:
            self.model = self._run_optimization()


    @abstractmethod
    def build_model(self) -> BaseEstimator:
        """Return a fresh (unfitted) sklearn-like estimator/pipeline."""
        raise NotImplementedError


    def optim_model_name(self) -> Optional[str]:
        """
        Return Optuna registry key (e.g., 'HuberSGD') if you want to use OptunaTuner.
        Subclasses can override.
        """
        return None


    def _run_optimization(self) -> BaseEstimator:
        from EvaluationToolbox import OptunaTuner

        name = self.optim_model_name()
        if not name:
            raise ValueError(
                f"{self.__class__.__name__} has use_optim = True but optim_model_name() returned None."
            )

        optimizer = OptunaTuner(
            metric = self.config.metric,
            n_splits = self.config.n_splits,
            random_state = self.config.seed,
        )
        optimizer.run(
            model_name = name,
            X = self.X_train, y = self.y_train
        )

        # Important: ensure the returned best_model_ is either already fitted, or we fit later in fit()
        return optimizer.best_model_


    def fit(self) -> "BaseTabularRegressor":
        self.model.fit(self.X_train, self.y_train)
        return self


    def predict(self) -> pd.Series:
        y_pred = self.model.predict(self.X_test)
        return pd.Series(y_pred, index = self.y_test.index, name = "y_pred")


    def evaluate(self) -> Dict[str, float]:
        from EvaluationToolbox import Evaluator

        y_pred = self.predict()
        evaluator = Evaluator(y_pred, self.y_test)
        return {
            "MSE": evaluator.score_MSE(),
            "RMSE": evaluator.score_RMSE(),
            "MAE": evaluator.score_MAE(),
            "Adjusted R2 score": evaluator.score_r2(
                use_adjusted=True, num_feat=self.X_train.shape[1]
            ),
        }


    def sketch_model(
        self,
        num_points: int = 200,
    ):
        from EvaluationToolbox import ModelSketcher

        """
        Creates a grid of plots like a pairplot, but each cell is:
        scatter: X[feat] vs y_true
        line: model prediction as X[feat] varies, other features fixed to center.
        """
        sketcher = ModelSketcher(self.model)
        sketcher.scatter_with_model_prediction(self.X_test, self.y_test, num_points = num_points)



class LinearRegressor(BaseTabularRegressor):
    def __init__(
        self,
        df_train: pd.DataFrame, df_test: pd.DataFrame,
        config: ModelConfig,
        epsilon = 1.35, alpha = 1e-4,
        max_iter = 5000, tol = 1e-4,
        l1_ratio = 0.2
    ):
        self.epsilon = epsilon
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.l1_ratio = l1_ratio
        self.config = config
        super().__init__(df_train, df_test, config) # return self.model


    def build_model(self):
        # Scale both X and y so the SGD step sizes stay stable across datasets
        return SGDRegressor(
                loss = "huber",
                penalty = "elasticnet",
                l1_ratio = self.l1_ratio,
                epsilon = self.epsilon,
                alpha = self.alpha,
                max_iter = self.max_iter,
                tol = self.tol,
                random_state = self.config.seed,
            )


    def optim_model_name(self) -> Optional[str]:
        # Must match the OptunaTuner registry key
        return "HuberSGD"



class PolynomialRegressor(BaseTabularRegressor):
    def __init__(
        self,
        df_train: pd.DataFrame, df_test: pd.DataFrame,
        config: ModelConfig,
        degree: int = 2,
        epsilon = 1.35, alpha = 5e-4,
        max_iter = 5000, tol = 1e-4,
        l1_ratio = 0.25
    ):
        self.degree = degree
        self.epsilon = epsilon
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.l1_ratio = l1_ratio
        super().__init__(df_train, df_test, config)


    def build_model(self):
        # Scale both X and y to stabilize SGD across datasets
        return make_pipeline(
                PolynomialFeatures(
                    degree = self.degree,
                    include_bias = False,
                    interaction_only = True, # reduces the impact of a single feature
                ),
                SGDRegressor(
                    loss = "huber",
                    penalty = "elasticnet",
                    epsilon = self.epsilon,
                    alpha = self.alpha,
                    max_iter = self.max_iter,
                    tol = self.tol,
                    l1_ratio = self.l1_ratio,
                    random_state = self.config.seed,
                )
            )


    def optim_model_name(self) -> Optional[str]:
        # Must match the OptunaTuner registry key
        return "PolySGD"



class KNNRegressor(BaseTabularRegressor):
    def __init__(
        self,
        df_train: pd.DataFrame, df_test: pd.DataFrame,
        config: ModelConfig,
        weights: str = "distance",     # uniform or distance
        n_neighbors: int = 5
    ):
        self.n_neighbors = n_neighbors
        self.weights = weights
        super().__init__(df_train, df_test, config)


    def build_model(self):
        return KNeighborsRegressor(
                n_neighbors = self.n_neighbors,
                weights = self.weights,
                algorithm = "auto"
            )


    def optim_model_name(self) -> Optional[str]:
        # Must match the OptunaTuner registry key
        return "KNNRegressor"



class SVMRegressor(BaseTabularRegressor):
    def __init__(
        self,
        df_train: pd.DataFrame, df_test: pd.DataFrame,
        config: ModelConfig,
        epsilon = 0.2, alpha = 1e-4,
        max_iter = 5000, tol = 1e-4,
    ):
        self.epsilon = epsilon
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        super().__init__(df_train, df_test, config)


    def build_model(self):
        # Scale both X and y to keep epsilon-insensitive SGD stable
        return SGDRegressor(
                loss = "epsilon_insensitive",
                epsilon = self.epsilon,
                alpha = self.alpha,
                max_iter = self.max_iter,
                tol = self.tol,
                penalty = "l2",
                learning_rate = "optimal",
                random_state = self.config.seed,
            )


    def optim_model_name(self) -> Optional[str]:
        # Must match the OptunaTuner registry key
        return "SVMRegressor"



class DTRegressor(BaseTabularRegressor):
    def __init__(
        self,
        df_train: pd.DataFrame, df_test: pd.DataFrame,
        config: ModelConfig,
        max_depth = 12, min_samples_leaf = 20,
        min_samples_split = 40, max_features = "sqrt"
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        super().__init__(df_train, df_test, config)


    def build_model(self):
        return DecisionTreeRegressor(
                random_state = self.config.seed,
                max_depth = self.max_depth,
                min_samples_leaf = self.min_samples_leaf,
                min_samples_split = self.min_samples_split,
                max_features = self.max_features,
            )


    def optim_model_name(self) -> Optional[str]:
        # Must match the OptunaTuner registry key
        return "DTRegressor"



class RFRegressor(BaseTabularRegressor):
    def __init__(
        self,
        df_train: pd.DataFrame, df_test: pd.DataFrame,
        config: ModelConfig,
        n_estimators = 400, max_samples = None, n_jobs = -1,
        max_depth = 12, min_samples_leaf = 20,
        min_samples_split = 40, max_features = "sqrt",
    ):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.n_jobs = n_jobs
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        super().__init__(df_train, df_test, config)


    def build_model(self):
        return RandomForestRegressor(
                random_state = self.config.seed,
                n_estimators = self.n_estimators,
                max_depth = self.max_depth,
                min_samples_leaf = self.min_samples_leaf,
                min_samples_split = self.min_samples_split,
                max_features = self.max_features,
                max_samples = self.max_samples,
                bootstrap = True,
                n_jobs = self.n_jobs
            )


    def optim_model_name(self) -> Optional[str]:
        # Must match the OptunaTuner registry key
        return "RFRegressor"



class GBRegressor(BaseTabularRegressor):
    def __init__(
        self,
        df_train: pd.DataFrame, df_test: pd.DataFrame,
        config: ModelConfig,
        n_estimators = 600,
        learning_rate = 0.05, max_depth = 3, subsample = 0.8,
        min_samples_leaf = 20, min_samples_split = 40,
        max_features = None,       # or "sqrt"/"log2"/float fraction
        tol = 1e-4
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.tol = tol
        super().__init__(df_train, df_test, config)


    def build_model(self):
        return GradientBoostingRegressor(
                random_state = self.config.seed,
                n_estimators = self.n_estimators,
                learning_rate = self.learning_rate,
                max_depth = self.max_depth,
                subsample = self.subsample,
                min_samples_leaf = self.min_samples_leaf,
                min_samples_split = self.min_samples_split,
                max_features = self.max_features,
                tol = self.tol,
            )


    def optim_model_name(self) -> Optional[str]:
        # Must match the OptunaTuner registry key
        return "GBRegressor"



class ABRegressor(BaseTabularRegressor):
    def __init__(
        self,
        df_train: pd.DataFrame, df_test: pd.DataFrame,
        config: ModelConfig,
        max_depth = 12, min_samples_leaf = 20,
        min_samples_split = 40, max_features = "sqrt",
        n_estimators = 400, learning_rate = 0.05,
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        super().__init__(df_train, df_test, config)


    def build_model(self):
        base_estimator = DecisionTreeRegressor(
            random_state = self.config.seed,
            max_depth = self.max_depth,
            min_samples_leaf = self.min_samples_leaf,
            min_samples_split = self.min_samples_split,
            max_features = self.max_features,
        )

        return AdaBoostRegressor(
                random_state = self.config.seed,
                estimator = base_estimator,
                n_estimators = self.n_estimators,
                learning_rate = self.learning_rate,
            )


    def optim_model_name(self) -> Optional[str]:
        # Must match the OptunaTuner registry key
        return "ABRegressor"



class XGBRegressor(BaseTabularRegressor):
    def __init__(
        self,
        df_train: pd.DataFrame, df_test: pd.DataFrame,
        config: ModelConfig,
        n_jobs = -1, n_estimators = 2000,
        learning_rate = 0.03, max_depth = 6,
        subsample = 0.8, colsample_bytree = 0.8,
        reg_lambda = 1.0, reg_alpha = 1.0, gamma = 1.0,
        max_bin = 64
    ):
        self.n_jobs = n_jobs
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.gamma = gamma
        self.max_bin = max_bin
        super().__init__(df_train, df_test, config)


    def build_model(self):
        return xgb.XGBRegressor(
                random_state = self.config.seed,
                n_jobs = self.n_jobs,
                max_depth = self.max_depth,
                n_estimators = self.n_estimators,
                learning_rate = self.learning_rate,
                subsample = self.subsample,
                colsample_bytree = self.colsample_bytree,
                reg_lambda = self.reg_lambda,
                reg_alpha = self.reg_alpha,
                gamma = self.gamma,
                max_bin = self.max_bin,
            ),


    def optim_model_name(self) -> Optional[str]:
        # Must match the OptunaTuner registry key
        return "XGBRegressor"



class LightGBMRegressor(BaseTabularRegressor):
    def __init__(
        self,
        df_train: pd.DataFrame, df_test: pd.DataFrame,
        config: ModelConfig,
        n_estimators = 2000,        # large; rely on early stopping
        learning_rate = 0.03,
        num_leaves = 63,              # for <20 features
        max_depth = -1,               # -1 = no limit
        min_child_samples = 50,       # a.k.a. min_data_in_leaf
        subsample = 0.8,              # bagging_fraction
        colsample_bytree = 0.8,       # feature_fraction
        reg_lambda = 1.0,
        reg_alpha = 0.0,
        # Speed / stability
        n_jobs = -1,
        # early_stopping_rounds = 2000
    ):
        self.n_jobs = n_jobs
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        # self.early_stopping_rounds = early_stopping_rounds
        super().__init__(df_train, df_test, config)


    def build_model(self):
        return LGBMRegressor(
                verbosity = -1,
                random_state = self.config.seed,
                n_jobs = self.n_jobs,
                max_depth = self.max_depth,
                n_estimators = self.n_estimators,
                learning_rate = self.learning_rate,
                subsample = self.subsample,
                colsample_bytree = self.colsample_bytree,
                reg_lambda = self.reg_lambda,
                reg_alpha = self.reg_alpha,
                num_leaves = self.num_leaves,
                min_child_samples = self.min_child_samples,
                # early_stopping_rounds = self.early_stopping_rounds,
            )


    def optim_model_name(self) -> Optional[str]:
            # Must match the OptunaTuner registry key
            return "LightGBMRegressor"


# ===================== DEEP LEARNING MODELS (support functions and classes) ==============================
def normalization(norm: str, dim: int):
    norm = norm.lower()
    match norm:
        case "batchnorm1d":
            return nn.BatchNorm1d(dim)
        case "layernorm":
            return nn.LayerNorm(dim)
        case "" | "none":
            return nn.Identity()
        case _:
            raise ValueError(f"Unknown norm: {norm}")


def activation_fn(act_name: str):
    act_name = act_name.lower()
    match act_name:
        case "relu":
            return nn.ReLU()
        case "gelu":
            return nn.GELU()
        case "sigmoid":
            return nn.Sigmoid()
        case "tanh":
            return nn.Tanh()
        case "silu" | "swish":
            return nn.SiLU()
        case _:
            raise ValueError(f"Unknown activation function: {act_name}")



class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]



class CreateDataLoader:
    def __init__(
        self,
        X_train, X_test,
        y_train, y_test,
        batch_size: int = 512,
        val_size: float = 0.1,
        is_shuffle = True,
        seed: int = 42,
    ):
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size = val_size,
            shuffle = is_shuffle,
            random_state = seed,
        )

        # Save
        self.X_train = self._to_tensor(X_train)
        self.X_val = self._to_tensor(X_val)

        # Do not convert X_test and y_test, keep it in pd.DataFrame to plot it later
        self.X_test_tensor = self._to_tensor(X_test)

        self.y_train = self._to_tensor(y_train)
        self.y_val   = self._to_tensor(y_val)

        self.batch_size = batch_size
        self.is_shuffle = is_shuffle


    def _to_tensor(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy(dtype=np.float32, copy=False)
        else:
            X = np.asarray(X, dtype=np.float32)

        if X.ndim == 1:
            X = X.reshape(-1)

        return torch.from_numpy(X)


    def create(self):
        train_dataset = CustomDataset(self.X_train, self.y_train)
        val_dataset = CustomDataset(self.X_val, self.y_val)
        train_loader = DataLoader(
            train_dataset,
            batch_size = self.batch_size,
            shuffle = self.is_shuffle,
            num_workers = 4,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size = self.batch_size,
            shuffle = not self.is_shuffle,
            num_workers = 4,
        )

        return train_loader, val_loader



class DeepTabularRegressor(BaseTabularRegressor):
    def __init__(
        self,
        df_train: pd.DataFrame, df_test: pd.DataFrame,
        config: ModelConfig,
        epochs: int = 100,
        batch_size: int = 2048,
        seed: int = 42,
        optim_present: bool = False
    ):
        self.device = pick_device()
        self.epochs = epochs
        self.batch_size = batch_size

        super().__init__(df_train, df_test, config) # self.model is built here
        self.model = self.model.to(self.device)

        self.num_feat = self.X_train.shape[1]
        self.criterion = nn.MSELoss()

        if not optim_present:
            self.optimizer = optim.Adam(self.model.parameters(),
                                lr = self.lr,
                                weight_decay = self.weight_decay
                            )

        if df_test is not None:
            DataloaderCreator = CreateDataLoader(
                self.X_train, self.X_test, self.y_train, self.y_test,
                batch_size = batch_size, seed = seed
            )
            self.train_loader, self.val_loader = DataloaderCreator.create()
            self.X_test_tensor = DataloaderCreator.X_test_tensor.to(self.device)


    def _to_numpy(self, x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)


    def _to_tensor(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy(dtype=np.float32, copy=False)
        else:
            X = np.asarray(X, dtype=np.float32)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return torch.from_numpy(X).to(self.device)


    def _score_r2(self, y_true, y_pred, use_adjusted = True, num_feat = None):
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)
        r2 = r2_score(y_true, y_pred)
        if use_adjusted:
            if num_feat is None:
                print("missing number of features")
                return None
            n = len(y_true)
            r2 = 1 - ((1 - r2) * ((n - 1) / (n - num_feat - 1)))
        return r2


    def get_params(self):
        return None


    def fit(
        self,
        X = None, y = None,
        plot_train_progress = False,
    ) -> "DeepTabularRegressor":

        train_losses = []
        val_losses = []
        train_adjusted_r2 = []
        val_adjusted_r2 = []

        if X is not None and y is not None:
            X = self._to_tensor(X)
            y = self._to_tensor(y)
            train_dataset = CustomDataset(X, y)
            self.train_loader = DataLoader(
                train_dataset,
                batch_size = self.batch_size,
                shuffle = True,
                num_workers = 0,
            )

        for epoch in range(self.epochs):
            train_loss = 0.0
            train_target = []
            train_predict = []

            # Training
            self.model.train()
            for X_samples, y_samples in self.train_loader:
                X_samples, y_samples = X_samples.to(self.device), y_samples.to(self.device)
                self.optimizer.zero_grad()
                y_preds = self.model(X_samples).view(-1)
                loss = self.criterion(y_preds, y_samples)
                loss.backward()
                self.optimizer.step()

                train_target.append(y_samples.detach())
                train_predict.append(y_preds.detach())
                train_loss += loss.item()

            train_loss /= len(self.train_loader)
            train_losses.append(train_loss)

            train_target_t = torch.cat(train_target, dim=0)
            train_predict_t = torch.cat(train_predict, dim=0)
            train_adjusted_r2.append(self._score_r2(
                y_true=train_target_t, y_pred=train_predict_t,
                use_adjusted=True, num_feat=self.num_feat
            ))

            # Skip if this is unecessary
            if plot_train_progress:
                # Evaluation
                val_loss = 0.0
                val_target = []
                val_predict = []

                self.model.eval()
                with torch.no_grad():
                    for X_samples, y_samples in self.val_loader:
                        X_samples, y_samples = X_samples.to(self.device), y_samples.to(self.device)
                        y_preds = self.model(X_samples)
                        loss = self.criterion(y_preds, y_samples)

                        val_loss += loss.item()
                        val_target.append(y_samples.detach())
                        val_predict.append(y_preds.detach())

                val_loss /= len(self.val_loader)
                val_losses.append(val_loss)

                val_target_t = torch.cat(val_target, dim=0)
                val_predict_t = torch.cat(val_predict, dim=0)
                val_adjusted_r2.append(self._score_r2(
                    y_true=val_target_t, y_pred=val_predict_t,
                    use_adjusted=True, num_feat=self.num_feat
                ))
                if (epoch + 1) % 10 == 0:
                    print(f'\nEPOCH {epoch + 1}:\tTraining loss: {train_loss:.3f}\tValidation loss: {val_loss:.3f}')

        if plot_train_progress:
            self._plot_train_progress(
                train_losses, val_losses,
                train_adjusted_r2, val_adjusted_r2
            )

        return self


    @torch.no_grad()
    def predict(self, X_test_input = None) -> pd.Series:
        self.model.eval()
        if X_test_input is None:
            X_test = getattr(self, "X_test_tensor", None)
            y_pred = self.model(X_test).detach().cpu().numpy()
            return pd.Series(y_pred, index = self.y_test.index, name = "y_pred")
        else:
            X_test = self._to_tensor(X_test_input)
            y_pred = self.model(X_test).detach().cpu().numpy()
            return y_pred


    def _plot_train_progress(
        self,
        train_losses, val_losses,
        train_r2, val_r2
    ):
        fig, ax = plt.subplots(2, 2, figsize=(12, 10))
        ax[0, 0].plot(train_losses, color='green')
        ax[0, 0].set(xlabel='Epoch', ylabel='Loss')
        ax[0, 0].set_title('Training Loss')
        ax[0, 0].grid(True, linestyle = "--", alpha = 0.5)

        ax[0, 1].plot(val_losses, color='orange')
        ax[0, 1].set(xlabel='Epoch', ylabel='Loss')
        ax[0, 1].set_title('Validation Loss')
        ax[0, 1].grid(True, linestyle = "--", alpha = 0.5)

        ax[1, 0].plot(train_r2, color='green')
        ax[1, 0].set(xlabel='Epoch', ylabel='R2')
        ax[1, 0].set_title('Training R2')
        ax[1, 0].grid(True, linestyle = "--", alpha = 0.5)

        ax[1, 1].plot(val_r2, color='orange')
        ax[1, 1].set(xlabel='Epoch', ylabel='R2')
        ax[1, 1].set_title('Validation R2')
        ax[1, 1].grid(True, linestyle = "--", alpha = 0.5)

        plt.show()



# ===================== DEEP LEARNING MODELS (Models implementation) ==============================


# ===================== RealMLP ==============================
class RealMLPRegressor(BaseTabularRegressor):
    def __init__(
        self,
        df_train: pd.DataFrame, df_test: pd.DataFrame,
        config: ModelConfig,
        n_epochs = 100, batch_size = 1024, predict_batch_size = 4096,
        n_hidden_layers = 4, hidden_width = 64,
        lr = 5e-2, wd = 1e-6, p_drop = 0.05
    ):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.predict_batch_size = predict_batch_size
        self.n_hidden_layers = n_hidden_layers
        self.hidden_width = hidden_width
        self.lr = lr
        self.wd = wd
        self.p_drop = p_drop
        super().__init__(df_train, df_test, config)


    def build_model(self):
        # Wrap with target scaling to keep loss scale consistent across datasets
        return RealMLP_TD_Regressor(
                device = pick_device(),
                random_state = self.config.seed,
                n_epochs = self.n_epochs,
                batch_size = self.batch_size,
                predict_batch_size = self.predict_batch_size,
                hidden_sizes = "rectangular",
                n_hidden_layers = self.n_hidden_layers,
                hidden_width = self.hidden_width,
                lr = self.lr,
                wd = self.wd,
                p_drop = self.p_drop,
                verbosity = 0,
                use_plr_embeddings = False, use_parametric_act = False, # for faster training
                act = "mish" # activation function for regression
            )


    def optim_model_name(self) -> Optional[str]:
        # Must match the OptunaTuner registry key
        return "RealMLPRegressor"
# ===================== RealMLP ==============================



# ===================== ResNet Regressor ==============================
class ResBlock(nn.Module):
    def __init__(
        self,
        d: int,
        d_hidden: int,
        act_fn: str = "relu",
        norm: str = "batchnorm1d",
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.norm = normalization(norm, d)
        self.fc1 = nn.Linear(d, d_hidden)
        self.activation = activation_fn(act_fn)
        self.drop = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(d_hidden, d)


    def forward(self, x):
        identity = x     # For Skip Connection
        x = self.norm(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.drop(x)
        x = x + identity  # Skip Connection

        return x



class ResNet(nn.Module):
    def __init__(
        self,
        df_train: pd.DataFrame, df_test: pd.DataFrame,
        config: ModelConfig,
        d_in: int,
        d: int = 256,
        n_res_blocks: int = 8,
        d_out: int = 1,
        d_hidden_factor: float = 0.2,
        dropout_rate: float = 0.2,
        act_fn: str = "relu",
        norm: str = "batchnorm1d",

        # Training hyperparams
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 100,
        batch_size: int = 1024
    ):
        super().__init__()
        d_hidden = int(d * d_hidden_factor)
        self.fc0 = nn.Linear(d_in, d)

        self.resnet_layers = nn.ModuleList(
            [
                ResBlock(
                    d = d,
                    d_hidden = d_hidden,
                    act_fn = act_fn,
                    norm = norm,
                    dropout_rate = dropout_rate
                )
                for _ in range(n_res_blocks)
            ]
        )

        self.prediction = nn.Sequential(
            normalization(norm, d),
            activation_fn(act_fn),
            nn.Linear(d, d_out)
        )

        self.param_dict = {
            "df_train": None, "df_test": None,
            "config": config,

            "d_in": d_in,
            "d": d,
            "n_res_blocks": n_res_blocks,
            "d_out": d_out,
            "d_hidden_factor": d_hidden_factor,
            "dropout_rate": dropout_rate,
            "act_fn": act_fn,
            "norm": norm,

            # Training hyperparams
            "lr": lr,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "batch_size": batch_size
        }


    def forward(self, x):
        x = self.fc0(x)
        for layer in self.resnet_layers:
            x = layer(x)
        x = self.prediction(x)
        return x.squeeze(-1)


    def get_params(self):
        return self.param_dict



class ResnetRegressor(DeepTabularRegressor):
    def __init__(
        self,
        df_train: pd.DataFrame, df_test: pd.DataFrame,
        config: ModelConfig,

        # ResNet
        random_state: int = 42,
        d_in: int = 1,
        d: int = 512,
        n_res_blocks: int = 8,
        d_out: int = 1,
        d_hidden_factor: float = 0.2,
        dropout_rate: float = 0.2,
        act_fn: str = "relu",
        norm: str = "batchnorm1d",

        # Training hyperparams
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 100,
        batch_size: int = 2048,
    ):
        self.param_dict = {
            "df_train": df_train, "df_test": df_test,
            "config": config,

            "d_in": d_in,
            "d": d,
            "n_res_blocks": n_res_blocks,
            "d_out": d_out,
            "d_hidden_factor": d_hidden_factor,
            "dropout_rate": dropout_rate,
            "act_fn": act_fn,
            "norm": norm,

            # Training hyperparams
            "lr": lr,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "batch_size": batch_size
        }

        self.lr = lr
        self.weight_decay = weight_decay
        super().__init__(df_train, df_test, config, epochs, batch_size, seed = random_state)


    def build_model(self):
        return ResNet(**self.param_dict)


    def optim_model_name(self) -> Optional[str]:
        # Must match the OptunaTuner registry key
        return "ResnetRegressor"
# ===================== ResNet Regressor ==============================



# ===================== FT-Transformer Regressor ==============================
class FTTransformerRegressor(DeepTabularRegressor):
    def __init__(
        self,
        df_train: pd.DataFrame, df_test: pd.DataFrame,
        config: ModelConfig,
        d_in: int, n_blocks: int = 3,
        epochs = 100, batch_size = 2048, seed = 42
    ):
        super().__init__(df_train, df_test, config, epochs, batch_size, seed, True)
        self.optimizer = self.model.make_default_optimizer()
        self.d_in = d_in
        self.n_blocks = n_blocks

        self.param_dict = {
            "df_train": df_train, "df_test": df_test,
            "config": config,

            "d_in": d_in,
            "n_blocks": n_blocks,
        }


    def build_model(self):
        kwargs = FTTransformer.get_default_kwargs(n_blocks = self.n_blocks)
        kwargs["d_out"] = 1
        model = FTTransformer(
            n_cont_features = self.d_in,
            cat_cardinalities = [],
            _is_default = True,
            **kwargs
        )
        return model


    def get_params(self):
        return self.param_dict


    def optim_model_name(self) -> Optional[str]:
        # Must match the OptunaTuner registry key
        return "FTTransformerRegressor"