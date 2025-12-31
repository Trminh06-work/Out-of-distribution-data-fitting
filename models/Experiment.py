import os
import numpy as np
import pandas as pd

import json
from tqdm.notebook import tqdm
from pathlib import Path
import gc
import copy
from collections import defaultdict

from sklearn.preprocessing import StandardScaler

from Models import ModelConfig, \
    LinearRegressor, PolynomialRegressor, KNNRegressor, SVMRegressor, \
    DTRegressor, RFRegressor, GBRegressor, ABRegressor, XGBRegressor, LightGBMRegressor, \
    RealMLPRegressor, ResnetRegressor, FTTransformerRegressor

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


MODEL_REGISTRY = {
    "LinearRegressor": LinearRegressor,
    "PolynomialRegressor": PolynomialRegressor,
    "KNNRegressor": KNNRegressor,
    "SVMRegressor": SVMRegressor,
    "DTRegressor": DTRegressor,
    "RFRegressor": RFRegressor,
    "GBRegressor": GBRegressor,
    "ABRegressor": ABRegressor,
    "XGBRegressor": XGBRegressor,
    "LightGBMRegressor": LightGBMRegressor,
    "RealMLPRegressor": RealMLPRegressor,
    "ResnetRegressor": ResnetRegressor,
    "FTTransformerRegressor": FTTransformerRegressor
}

DATASET_LIST = [
    "218_house_8L", "house", "california", "elevators", "CASP", "gas_turbine_co_and_nox_emission",
    "CBM_1", "CBM_2", "delays_zurich_transport", "diamonds", "nyc_taxi_green_dec_2016", "house_sales", "medical_charges", 
    "MiamiHousing2016", "sulfur_1", "sulfur_2", "books", "players_22", "taxi", "bike",
    "power_consumption", "3droad", "keggdirected", "kin40k", "protein", "tamielectric"
]

SPLIT_TYPES = [
    "Random_Split", "Covariate_Shift", "Mfs_based_Split",
    "Single_Hyperball", "Multiple_Hyperballs", "KMeans_Hyperballs",
    "Single_Slab", "Semi_Infinite_Slab"
]



class DataSaver:
    def __init__(self, model_name):
        self.model_name = model_name
        self.output_dir = "Results/"
        os.makedirs(self.output_dir, exist_ok = True)


    def _to_python(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if hasattr(obj, "item"):   # torch / numpy scalar
            return obj.item()
        raise TypeError


    def save_result(self, out_file, results):
        # Save path:  Results/{model_name}/{file_name}.json
        # where in each .json file, it encompasses of results for all types of split
        try:
            with open(out_file, "w") as f:
                json.dump(results, f, indent = 2, default = self._to_python)
            # tqdm.write(f"Successfully saved → {out_file}")
        except:
            tqdm.write(f"Error: Cannot save file")



class EvaluateModel:
    def __init__(self, config: ModelConfig, model_name: str = None):
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_name}")
        self.model_name = model_name
        self.model_class = MODEL_REGISTRY[model_name]
        self.config = config


    def _process(self, df_train, df_test):
        config = copy.deepcopy(self.config)
        if self.model_name in list(MODEL_REGISTRY.keys())[-3:]:
            config.use_optim = False

        if self.model_class is ResnetRegressor:
            regressor = ResnetRegressor(df_train, df_test, config, d_in = df_train.shape[1] - 1)
        elif self.model_class is FTTransformerRegressor:
            regressor = FTTransformerRegressor(df_train, df_test, config, d_in = df_train.shape[1] - 1, n_blocks = 4)
        else:
            regressor = self.model_class(df_train, df_test, config)

        regressor.fit()
        result = regressor.evaluate()

        return result


    def evaluate(self, ds_lst = DATASET_LIST):
        path = f"../data/splitted"
        data_saver = DataSaver(self.model_name)

        for file_name in tqdm(ds_lst, desc = "Dataset processing"):
            results = defaultdict(dict)

            save_dir = os.path.join("Results/", self.model_name)
            os.makedirs(save_dir, exist_ok=True)
            out_file = os.path.join(save_dir, f"{file_name}.json")
            if os.path.exists(out_file):
                tqdm.write("Skip due to file exists")
                continue

            for split_type in tqdm(SPLIT_TYPES, desc = f"Processing {file_name} splits", leave = False):
                folder = Path(os.path.join(path, file_name, split_type))
                train_files = sorted(folder.glob("train_*.parquet"))

                for train_file in tqdm(train_files, desc = f"{file_name}/{split_type}", leave = False):
                    idx = train_file.stem.split("_")[1]
                    test_file = folder / f"test_{idx}.parquet"

                    if not test_file.exists():
                        tqdm.write(f"Warning: test file missing for idx={idx}")
                        continue

                    try:
                        scaler = StandardScaler()
                        df_train = pd.read_parquet(train_file)
                        df_test = pd.read_parquet(test_file)

                        df_train = pd.DataFrame(
                            scaler.fit_transform(df_train),
                            columns = df_train.columns
                        )
                        df_test = pd.DataFrame(
                            scaler.transform(df_test),
                            columns = df_test.columns
                        )
                    except Exception as e:
                        tqdm.write(f"Read failed for idx = {idx}: {e}")
                        continue

                    results[split_type][idx] = self._process(df_train, df_test)

                    del df_train, df_test
                    gc.collect()

            data_saver.save_result(out_file, results)


