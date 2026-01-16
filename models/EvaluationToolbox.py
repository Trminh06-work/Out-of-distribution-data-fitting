import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.pipeline import make_pipeline
from sklearn.base import clone, BaseEstimator
from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
import xgboost as xgb
from lightgbm import LGBMRegressor

from pytabkit import RealMLP_TD_Regressor
from ft_transformer import FTTransformer
from Models import ResNet, ResnetRegressor, FTTransformerRegressor, pick_device, ModelConfig

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


import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING) # Stop logging output
from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional



class Evaluator:
    def __init__(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true


    def score_MSE(self):
        rmse = mean_squared_error(self.y_true, self.y_pred)
        return rmse


    def score_RMSE(self):
        rmse = np.sqrt(mean_squared_error(self.y_true, self.y_pred))
        return rmse


    def score_MAE(self):
        mae = mean_absolute_error(self.y_true, self.y_pred)
        return mae


    def score_r2(self, use_adjusted = False, num_feat = None):
        r2 = r2_score(self.y_true, self.y_pred)
        if use_adjusted:
            if num_feat is None:
                print("missing number of features")
                return None
            n = len(self.y_true)
            r2 = 1 - ((1 - r2) * ((n - 1) / (n - num_feat - 1)))
        return r2


    def score_MAPE(self):
        mape = mean_absolute_percentage_error(self.y_true, self.y_pred)
        return mape


    def score_sMAPE(self):
        y_true = np.asarray(self.y_true, dtype=float).ravel()
        y_pred = np.asarray(self.y_pred, dtype=float).ravel()

        epsilon = 1e-4 # avoid zero division
        denom = np.abs(y_true) + np.abs(y_pred) + epsilon
        num = np.abs(y_true - y_pred)
        sMAPE = 200.0 * np.mean(num / denom)

        return sMAPE


    def score_nRMSE(self):
        rmse = self.score_RMSE()
        return rmse / self.y_true.std()


    def score_nMAE(self):
        mae = self.score_MAE()
        return mae / self.y_true.std()


class ModelSketcher:
    def __init__(self, model):
        self.model = model


    def _construct_conf_band(
        self,
        X_grid: pd.DataFrame,
        X: pd.DataFrame,
        y_true: pd.Series,
        feat: str,
        ci: float = 95, # confidence level
        num_points: int = 300,
        n_boots: int = 50,
        random_state: int = 42,
    ):
        def pdp_curve(model, X_full, feat, x_grid):
            """
                Make predictions using new re-trained model on an interval "x_grid"
            """
            out = np.empty(len(x_grid), dtype=float)

            for i, g in enumerate(x_grid):
                # ALWAYS start from fresh copy
                X_tmp = X_full.copy()
                X_tmp[feat] = g      # vary ONE feature and keep other features fixed

                preds = model.predict(X_tmp)
                out[i] = np.mean(preds)     # average over all rows

            return out


        def model_factory(X_b, y_b):
            if isinstance(self.model, ResNet):
                param_dict = self.model.get_params()
                df_train = pd.concat([X_b, y_b], axis = 1)
                param_dict["df_train"] = df_train
                return ResnetRegressor(**param_dict)
            elif isinstance(self.model, FTTransformer):
                param_dict = self.model.get_params()
                df_train = pd.concat([X_b, y_b], axis = 1)
                param_dict["df_train"] = df_train
                return FTTransformerRegressor(**param_dict)
            else:
                return clone(self.model)


        boot_preds = []

        rng = np.random.default_rng(random_state)
        n = len(X)

        for _ in range(n_boots):
            idx = rng.integers(low = 0, high = n, size = num_points)       # sample rows with replacement
            X_b, y_b = X.iloc[idx], y_true.iloc[idx]

            m = model_factory(X_b, y_b)
            m.fit(X_b, y_b)

            boot_preds.append(pdp_curve(m, X, feat, X_grid))  # PDP evaluated on full X

        boot_preds = np.array(boot_preds)

        alpha = (100 - ci) / 2 # significance level
        lower = np.percentile(boot_preds, alpha, axis=0)
        upper = np.percentile(boot_preds, ci + alpha, axis=0)
        mean  = boot_preds.mean(axis=0)

        return lower, upper, mean


    def scatter_with_model_prediction(
        self,
        X: pd.DataFrame,
        y_true: pd.Series,
        keep_size: bool = False,
        random_state: int = 42,
        num_points: int = 300,
        figsize = (16, 10),
        n_cols: int = 3,            # number of subplot columns, defalut: -1 -> # cols = # features
    ):
        """
        Creates a grid of plots like a pairplot, but each cell is:
        scatter: X[feat] vs y_true
        line: model prediction as X[feat] varies, other features fixed to center.
        """
        # If plot all points
        if keep_size:
            num_points = len(y_true)

        if n_cols == -1:
            n_cols = X.shape[1]

        n_rows = int(np.ceil(X.shape[1] / n_cols))
        # Ensure aligned indices & clean arrays
        idx = X.sample(n = num_points, random_state = random_state).index

        X = X.loc[idx].copy().reset_index(drop=True)
        y_true = y_true.loc[idx].reset_index(drop=True)

        # Scale X
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(
            X_scaled,
            columns = X.columns,
            index = X.index
        )

        # Create subplots
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize = figsize,
            sharey = True
        )

        for j, (feat, ax) in enumerate(zip(X.columns, axes.flat)):
            sns.scatterplot(x = X[feat], y = y_true, alpha = 0.5, ax = ax)

            X_grid = np.linspace(X[feat].min(), X[feat].max(), 100)

            lower, upper, mean = self._construct_conf_band(
                X = X,
                y_true = y_true,
                feat = feat,
                X_grid = X_grid,
                n_boots = 25,
                ci = 95, # 95% Confidence band
                random_state = random_state
            )
        
            ax.plot(X_grid, mean, color="red", label="Mean prediction")
            ax.fill_between(
                X_grid,
                lower,
                upper,
                alpha=0.3,
                label="95% Confidence band",
                color = "orange"
            )

            ax.grid(True, linestyle = "--", alpha = 0.5)
            ymin = y_true.min()
            ymax = y_true.max()
            margin = 0.05 * (ymax - ymin) if ymax > ymin else 1.0
        
            ax.set_ylim(ymin - margin, ymax + margin)

            # reduce label clutter: y-label only on left column
            if (j % n_cols) != 0:
                ax.set_ylabel("")

        # Hide unused axes
        for ax in axes[len(X.columns):]:
            ax.set_visible(False)
        
        # Global legend
        legend_handles = [
            Line2D([], [], marker="o", linestyle="None", color="C0", alpha=0.5, label="Observed"),
            Line2D([], [], color="red", lw=2, label="PDP mean"),
            Patch(facecolor="orange", alpha=0.3, label="95% confidence band"),
        ]
        
        fig.legend(
            handles=legend_handles,
            loc="center right",
            # ncol=len(legend_handles),
            frameon=False,
            bbox_to_anchor=(1.01, 0.5)   # slightly outside, vertically centered
        )
        fig.suptitle(
            "Feature-wise Model Effects with 95% Confidence Intervals",
            fontsize=16,
            y=0.94   # move title up/down if needed
        )

        plt.tight_layout(rect=[0, 0, 0.88, 0.92])
        plt.show()


# Hyperparamets Search

# Linear Regression with Huber Loss params search space
def huber_sgd_space(trial):
    return {
        "epsilon": trial.suggest_float("epsilon", 1.05, 3.0),
        "alpha": trial.suggest_float("alpha", 1e-7, 1e-2, log=True),
        "max_iter": trial.suggest_int("max_iter", 1000, 20000, log=True),
        "tol": trial.suggest_float("tol", 1e-6, 1e-3, log=True),
        "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
    }


def huber_sgd_build(params, seed):
    return SGDRegressor(loss="huber", random_state = seed, **params)


# Polynomial Regression with Huber loss + L1-Regularization using ElasticNet
def poly_sgd_space(trial):
    return {
        "polynomialfeatures__degree": trial.suggest_int("polynomialfeatures__degree", 1, 4),

        "sgdregressor__epsilon": trial.suggest_float("sgdregressor__epsilon", 1.05, 3.0),
        "sgdregressor__alpha": trial.suggest_float("sgdregressor__alpha", 1e-7, 1e-2, log=True),
        "sgdregressor__l1_ratio": trial.suggest_float("sgdregressor__l1_ratio", 0.0, 1.0),
        "sgdregressor__max_iter": trial.suggest_int("sgdregressor__max_iter", 1000, 20000, log=True),
        "sgdregressor__tol": trial.suggest_float("sgdregressor__tol", 1e-6, 1e-3, log=True),
    }


def poly_sgd_build(params, seed):
    model =  make_pipeline(
        PolynomialFeatures(include_bias = False),
        SGDRegressor(loss = "huber", random_state = seed, penalty = "elasticnet")
    )
    model.set_params(**params)
    return model


# K-Neareast Neighbours Regressor
def knn_reg_space(trial):
    return {
        "n_neighbors": trial.suggest_int("n_neighbors", 1, 10),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"])
    }


def knn_reg_build(params, seed):
    return KNeighborsRegressor(**params)


# Support Vector Machine Regressor
def svm_reg_space(trial):
    return {
        "epsilon": trial.suggest_float("epsilon", 1.05, 3.0),
        "alpha": trial.suggest_float("alpha", 1e-7, 1e-2, log=True),
        "max_iter": trial.suggest_int("max_iter", 1000, 20000, log=True),
        "tol": trial.suggest_float("tol", 1e-6, 1e-3, log=True),
    }


def svm_reg_build(params, seed):
    return SGDRegressor(loss = "epsilon_insensitive", random_state = seed, **params)


# Decision Tree Regressor
def dt_reg_space(trial):
    return {
        "max_depth": trial.suggest_int("max_depth", 2, 30),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 200, log=True),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 500, log=True),
        "max_features": trial.suggest_categorical("max_features", [None, "sqrt", "log2"]),
    }


def dt_reg_build(params, seed):
    return DecisionTreeRegressor(random_state = seed, **params)


# Random Forest Regressor
def rf_reg_space(trial):
    return {
        "max_depth": trial.suggest_int("max_depth", 1, 20),  # keep shallow
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 200, log=True),
        "min_samples_split": trial.suggest_int("min_samples_split", 10, 400, log=True),
    
        # feature subsampling
        "max_features": trial.suggest_categorical(
            "max_features",
            ["sqrt", "log2", 0.3, 0.5, 0.7, 1.0]  # float = fraction of features
        ),
    
        # number of trees: moderate (big N)
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
    
        # subsample rows per tree (major speed lever)
        "max_samples": trial.suggest_float("max_samples", 0.4, 0.9),
    }


def rf_reg_build(params, seed):
    return RandomForestRegressor(random_state = seed, **params)


# Gradient Boosting Regressor
def gb_reg_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),

        # Tree shape (most important for generalization)
        "max_depth": trial.suggest_int("max_depth", 1, 6),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 300, log=True),
        "min_samples_split": trial.suggest_int("min_samples_split", 10, 800, log=True),

        # Stochastic GB for speed + regularization
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),

        # Feature subsampling (useful even with <20 features)
        "max_features": trial.suggest_categorical(
            "max_features",
            ["sqrt", "log2", 0.5, 0.7, 1.0]
        ),
    }


def gb_reg_build(params, seed):
    return GradientBoostingRegressor(random_state = seed, **params)


# AdaBoost Regressor
def ab_reg_space(trial):
    # Base tree (controls smoothness / OOD stability)
    max_depth = trial.suggest_int("max_depth", 1, 6)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 5, 300, log=True)
    min_samples_split = trial.suggest_int("min_samples_split", 10, 800, log=True)

    max_features = trial.suggest_categorical(
        "max_features",
        ["sqrt", "log2", 0.5, 0.7, 1.0]  # float = fraction of features
    )

    # AdaBoost knobs
    n_estimators = trial.suggest_int("n_estimators", 100, 1500)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.5, log=True)

    return {
        # Base tree
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "min_samples_split": min_samples_split,
        "max_features": max_features,

        # AdaBoost
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
    }


def ab_reg_build(params, seed):
    base_estimator = DecisionTreeRegressor(
        random_state = seed,
        max_depth = params["max_depth"],
        min_samples_leaf = params["min_samples_leaf"],
        min_samples_split = params["min_samples_split"],
        max_features = params["max_features"],
    )

    return AdaBoostRegressor(
            random_state = seed,
            estimator = base_estimator,
            n_estimators = params["n_estimators"],
            learning_rate = params["learning_rate"],
        )


# XGBoost Regressor
def xgb_reg_space(trial):
    return {
        # boosting
        "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),

        # tree shape (controls speed + generalization)
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 128.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 10.0),

        # regularization
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 1.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-2, 1.0, log=True),

        # subsampling (often improves OOD robustness + reduces compute a bit)
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),

        # hist speed/accuracy knob (only for hist/approx; higher = slower but can improve split quality)
        "max_bin": trial.suggest_categorical("max_bin", [64, 128, 256]),
    }


def xgb_reg_build(params, seed):
    return xgb.XGBRegressor(random_state = seed, n_jobs = -1, **params)


# LightGBM Regressor
def lightgbm_reg_space(trial):
    params = {
        # Core
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": "rmse",

        # Speed on CPU
        "n_jobs": -1,
        "verbosity": -1,

        # Let early stopping pick the effective number of trees
        "n_estimators": trial.suggest_int("n_estimators", 1000, 2000),
        "num_leaves": trial.suggest_int("num_leaves", 16, 256, log=True),

        # Main knobs
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),

        # Tree complexity (cap for speed + OOD stability)
        "max_depth": trial.suggest_int("max_depth", 3, 8),

        # Leaf / split regularization (very important for extrapolation stability)
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 5000, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),

        # Column / row sampling (regularization + sometimes speed)
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "subsample_freq": trial.suggest_int("subsample_freq", 1, 10),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),

        # L1/L2 regularization
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 1.0, log=True),

        # Histogram bins: big speed lever on CPU
        "max_bin": trial.suggest_categorical("max_bin", [63, 127, 255]),
    }

    # Consistency constraint: num_leaves should not exceed 2^max_depth too much
    # (not required, but helps avoid pointless combos)
    params["num_leaves"] = min(params["num_leaves"], 2 ** params["max_depth"])

    return params


def lightgbm_reg_build(params, seed):
    return LGBMRegressor(random_state = seed, **params)


# RealMLP Regressor
def realmlp_reg_space(trial):
    return {
        # budget
        "device": pick_device(),
        "n_epochs": 64,
        "batch_size": 1024,
        "predict_batch_size": 4096,

        # architecture (rectangular means hidden_width * n_hidden_layers)
        "hidden_sizes": "rectangular",
        "n_hidden_layers": trial.suggest_int("n_hidden_layers", 2, 5),
        "hidden_width": trial.suggest_categorical("hidden_width", [64, 128, 256]),

        # optimization / regularization
        "lr": trial.suggest_float("lr", 5e-2, 3e-1, log=True),
        "wd": trial.suggest_float("wd", 1e-6, 5e-2, log=True),
        "p_drop": trial.suggest_float("p_drop", 0.05, 0.35),
    }


def realmlp_reg_build(params, seed):
    return RealMLP_TD_Regressor(random_state = seed, verbosity = 0, **params)


# ResNet Regressor
def resnet_reg_space(trial):
    return {
        "df_train": None,
        "df_test": None,
        "d_in": None,
        "config": ModelConfig(use_optim = False),
        "d": trial.suggest_int("d", 64, 512),
        "n_res_blocks": trial.suggest_int("n_res_blocks", 1, 4),
        "d_out": 1,
        "d_hidden_factor": trial.suggest_float("d_hidden_factor", 0, 1),
        "dropout_rate": trial.suggest_float("dropout_rate", 0, 0.5),
        "act_fn": "relu",
        "norm": "batchnorm1d",
        "lr": trial.suggest_float("lr", 1e-5, 1e-2),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512, 1024]),
    }


def resnet_reg_build(params, seed):
    return ResnetRegressor(random_state = seed, **params)


# FT-Transformer Regressor
def ft_transformer_reg_space(trial):
    # The original code just allows to modify n_blocks in [1, 6]
    return {
        "df_train": None,
        "df_test": None,
        "d_in": None,
        "config": ModelConfig(use_optim = False),
        "n_blocks": trial.suggest_int("n_blocks", 1, 6),
    }


def ft_transformer_reg_build(params, seed):
    return FTTransformerRegressor(**params)



# Optuna Tuner setup
SearchSpaceFn = Callable[[optuna.trial.Trial], Dict[str, Any]]
BuildModelFn = Callable[[Dict[str, Any], int, int], BaseEstimator]


@dataclass
class ModelSpec:
    space: SearchSpaceFn
    build: BuildModelFn



class OptunaTuner:
    """
    Model-agnostic Optuna tuner for sklearn-like estimators.
    - Supports multiple model specs via registry.
    - CV-based objective for fair comparison.
    """
    def __init__(
        self,
        metric: str = "rmse",
        n_splits: int = 5,
        random_state: int = 42,
        direction: str = "minimize"
    ):
        self.registry = {
            "HuberSGD": ModelSpec(build = huber_sgd_build, space = huber_sgd_space),
            "PolySGD": ModelSpec(build = poly_sgd_build, space = poly_sgd_space),
            "KNNRegressor": ModelSpec(build = knn_reg_build, space = knn_reg_space),
            "SVMRegressor": ModelSpec(build = svm_reg_build, space = svm_reg_space),
            "DTRegressor": ModelSpec(build = dt_reg_build, space = dt_reg_space),
            "RFRegressor": ModelSpec(build = rf_reg_build, space = rf_reg_space),
            "GBRegressor": ModelSpec(build = gb_reg_build, space = gb_reg_space),
            "ABRegressor": ModelSpec(build = ab_reg_build, space = ab_reg_space),
            "XGBRegressor": ModelSpec(build = xgb_reg_build, space = xgb_reg_space),
            "LightGBMRegressor": ModelSpec(build = lightgbm_reg_build, space = lightgbm_reg_space),
            "RealMLPRegressor": ModelSpec(build = realmlp_reg_build, space = realmlp_reg_space),
            "ResnetRegressor": ModelSpec(build = resnet_reg_build, space = resnet_reg_space),
            "FTTransformerRegressor": ModelSpec(build = ft_transformer_reg_build, space = ft_transformer_reg_space)
        }
        self.metric = metric
        self.n_splits = n_splits
        self.random_state = random_state
        self.direction = direction

        self.study: Optional[optuna.Study] = None
        self.best_params_: Optional[Dict[str, Any]] = None
        self.best_value_: Optional[float] = None
        self.best_model_: Optional[BaseEstimator] = None


    def _score(self, y_true, y_pred):
        if self.metric == "rmse":
            return np.sqrt(mean_squared_error(y_true, y_pred))
        raise ValueError(f"Unsupported metric: {self.metric}")


    def run(
        self,
        model_name: str,
        X,
        y,
        n_trials: int = 5,
        timeout: Optional[int] = None,
        sampler: Optional[optuna.samplers.BaseSampler] = None,
    ):
        if model_name not in self.registry.keys():
            raise KeyError(f"Unknown model_name = '{model_name}'. Available: {list(self.registry)}")

        spec = self.registry[model_name]
        sampler = sampler or optuna.samplers.TPESampler(seed = self.random_state)

        def objective(trial: optuna.trial.Trial):
            params = spec.space(trial)
            cv = KFold(
                n_splits = self.n_splits,
                random_state = self.random_state,
                shuffle = True
            )

            scores = []
            for fold, (train_idx, val_idx) in enumerate(cv.split(X), start = 1):
                X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
                X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

                if model_name in ["ResnetRegressor", "FTTransformerRegressor"]:
                    params["df_train"] = pd.concat([X_train, y_train], axis = 1)
                    params["df_test"] = pd.concat([X_val, y_val], axis = 1)
                    params["d_in"] = X_train.shape[1]

                model = spec.build(params, self.random_state)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                score = self._score(y_val, y_pred)
                scores.append(score)

            return float(np.mean(scores))

        self.study = optuna.create_study(
            direction = self.direction,
            sampler = sampler,
            storage="sqlite:///optuna.db",
            load_if_exists=True
        )
        self.study.optimize(
            objective,
            n_trials = n_trials,
            timeout = timeout,
            n_jobs=6,
        )
        self.best_params_ = dict(self.study.best_params)
        self.best_value_ = float(self.study.best_value)

        # Build the model on the best params
        self.best_model_ = spec.build(self.best_params_, self.random_state)

        return self