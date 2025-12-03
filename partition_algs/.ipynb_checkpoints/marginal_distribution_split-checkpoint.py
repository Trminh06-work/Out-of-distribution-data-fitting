import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

class MarginalDistributionSplit:
    def __init__(self, file_name, df, test_size):
        self.file_name = file_name
        self.X = df.iloc[:, :-1]
        self.y = df.iloc[:, -1]
        self.test_size = test_size
        self.train_size = 1 - test_size
        self.SEEDS = [0, 42, 19, 2, 23, 11, 37, 29, 57, 5] # 10 random SEEDs to construct CI

    """
    Supportive function:
        _distribution_based_selection(self, feat)
    """

    def _distribution_based_selection(self, feat):
        """
        This function returns a permutatation of which rows are selected
        
        Arguments:
        X         :    the feature matrix
        test_size :    the proportion of train/test split
        feat      :    the feature column being considered
        
        Return:
        col_mask  :    a permuation of selected samples' indices
        """
        
        lo = 0
        hi = 1
        epsilon = 0.001
        
        while lo < hi:
            mid = lo + (hi - lo) / 2
            
            digitized_col = np.digitize(self.X.loc[:, feat], np.quantile(self.X.loc[:, feat], [mid, 1 - mid]))
            
            # col_mask keeps the bits 1 on the row that a specific column is numbered 1, otherwise 0.
            col_mask = (digitized_col == 1)
            
            if col_mask.sum() / self.X.shape[0] < self.test_size:
                lo = mid + epsilon
            else:
                hi = mid - epsilon
        
        return col_mask


    def covariate_prior_shift(self):
        """
        This function returns a list of permuation of selected samples w.r.t each feature.
        
        Arguments:
        X         :    the feature matrix
        test_size :    the proportion of train/test split
        
        Return:
        perm      :    a permuation of selected samples' indices
        """
    
        for idx, feat in range(self.X.shape[1] + 1): # + 1 for the target column
            mask = distribution_based_selection(feat)
            X_train = self.X[mask]
            y_train = self.y[mask]
            X_test = self.X[~mask]
            y_test = self.y[~mask]

            df_train = pd.concat([X_train, y_train], axis = 1)
            df_test = pd.concat([X_test, y_test], axis = 1)
    
            # Save files using the idx
            path = f"../data/splitted/{self.file_name}/Covariate_Prior_Shift/train_{idx}.csv"
            df_train.to_csv(path, index = False)
            path = f"../data/splitted/{self.file_name}/Covariate_Prior_Shift/test_{idx}.csv"
            df_test.to_csv(path, index = False)
        

