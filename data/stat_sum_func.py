import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from collections import Counter

class ToParquet:
    """
    This class is a tool to efficiently convert .csv/.tsv files to .parquet files.
    In particular, it converts files that are residing in child folders in the current base_dir folder.

    base_dir
    |
    |--- folder_1/
    |        |--- data_1.csv
    |        |--- data_2.pkl
    |        |--- data_3.tsv
    |
    |--- folder_2/
    |        |--- data_1.csv
    |        |--- data_2.tsv
    |        |--- data_3.parquet

    
    Arguments:
        base_dir:   the current folder
    """
    def __init__(self, base_dir):
        self.base_dir = base_dir


    # Function to read data files with error handling
    def _read_data_file(self, file_path):
        try:
            if file_path.endswith('.csv'):
                return pd.read_csv(file_path)
            elif file_path.endswith('.tsv'):
                return pd.read_csv(file_path, sep='\t')
            return None
        except Exception as e:
            # logging.error(f"Error reading {file_path}: {str(e)}")
            return None

    
    def main(self):
        # Loop through all folders in the base directory
        error_folders = []
        all_data = {}
        for folder_name in os.listdir(self.base_dir):
            try:
                folder_path = os.path.join(self.base_dir, folder_name)
                
                # Skip .ipynb_checkpoints directory
                if folder_name == '.ipynb_checkpoints':
                    continue
                
                # Check if it's a directory
                if os.path.isdir(folder_path):
                    # Try CSV first
                    csv_path = os.path.join(folder_path, f"{folder_name}.csv")
                    tsv_path = os.path.join(folder_path, f"{folder_name}.tsv")
                    
                    # Try to read the CSV file
                    df = self._read_data_file(csv_path)
                    
                    # If CSV doesn't exist or couldn't be read, try TSV
                    if df is None:
                        df = self._read_data_file(tsv_path)
                    
                    # If we successfully loaded data, add it to our dictionary
                    if df is not None:
                        all_data[folder_name] = df
                        print(f"Successfully loaded data from {folder_name}.")
                        new_path = os.path.join(folder_path, f"{folder_name}.parquet")
            
                        if os.path.exists(csv_path):
                            os.remove(csv_path)
                            print(f"{csv_path} file deleted.")
                        else:
                            print(f"{csv_path} file not found.")
            
                        if os.path.exists(tsv_path):
                            os.remove(tsv_path)
                            print(f"{tsv_path} file deleted.")
                        else:
                            print(f"{tsv_path} file not found.")
                        
                        print(f"Successfully saved data as {folder_name}.parquet")
                        df.to_parquet(new_path, index = False)
                        
                    else:
                        print(f"No data file found for {folder_name}")
                        error_folders.append(folder_name)
            except Exception as e:
                # Log the error and continue with the next folder
                error_message = f"Error processing folder {folder_name}: {str(e)}"
                print(error_message)
                # logging.error(error_message)
                error_folders.append(folder_name)
            
            print("===" * 20)
        
        # Now all_data contains dataframes for each folder that had a valid CSV or TSV file
        print(f"Loaded data from {len(all_data)} folders.")
        print(f"Encountered errors in {len(error_folders)} folders:")
        for error_folder in error_folders:
            print(error_folder)



class DatasetStatistics:
    def __init__(self, file_path):
        self.df = pd.read_parquet(file_path)
        self.X = self.df.iloc[:, :-1]
        self.y = self.df.iloc[:, -1]

    
    def plot_distribution(self, feat):
        """
        This function plots the distribution of values w.r.t each feature
        
        Arguments:
            feat -> str: the feature's name
        """
        plt.figure(figsize=(8, 5))
        
        sns.histplot(self.df[feat], kde = True, 
                     bins = 50, fill = False, 
                     stat = "density",
                    )
        plt.grid(True, linestyle = "--", alpha = 0.6)
        plt.title(f"Distribution of {feat}", fontsize = 14, weight = "bold")
        plt.tight_layout()
        plt.show()


    def plot_box(self):
        df_scaled = pd.DataFrame(
            StandardScaler().fit_transform(self.df),
            columns = self.df.columns
        )
        
        sns.boxplot(data = df_scaled, orient = "h")
        plt.xlabel("Standardized Value (z-score)")
        plt.title("Box Plot of Standardized Features" , fontsize = 14, weight = "bold")
        plt.grid(True, linestyle = "--", alpha = 0.6)
        plt.show()


    def print_stat_sum(self):
        print(f"Number of samples : {self.df.shape[0]}")
        print(f"Number of features: {self.df.shape[1] - 1}")
        print("==" * 15)
        desc = self.df.describe().T
        stats = pd.concat(
            [self.df.dtypes, self.df.isna().sum(), desc.iloc[:, 0], self.df.median(), desc.iloc[:, 1:]],
            axis = 1
        )
        stats = stats.rename(columns = {
                              0: "dtype", 
                              1: "missing",
                              2: "median"
        })
        display(stats)


    def plot_corr_heatmap(self):
        target = self.df.columns[-1]

        plt.figure(figsize=(10, 8))
        
        ax = sns.heatmap(
            self.df.corr(),
            cmap = "coolwarm",          # use cmap, not color
            annot = True,               # show correlation values
            fmt = ".2f",                # 2 decimal places
            linewidths = 0.5,
            square = True,
            vmin = -1, vmax = 1,
            cbar_kws = {"shrink": 0.8, "label": "Pearson's correlation"}
        )
        
        # Bold target on x-axis
        for label in ax.get_xticklabels():
            if label.get_text() == target:
                label.set_fontweight("bold")
        
        # Bold target on y-axis
        for label in ax.get_yticklabels():
            if label.get_text() == target:
                label.set_fontweight("bold")
        
        plt.title("Correlation Between Features and Target", fontsize = 14, weight = "bold")
        plt.xticks(rotation = 45, ha = "right")
        plt.yticks(rotation = 0)
        plt.tight_layout()
        plt.show()


class SplittedDatasetStatistics:
    def __init__(self, train_path, test_path):
        self.train = DatasetStatistics(train_path)
        self.test = DatasetStatistics(test_path)

        df_train = self.train.df.copy()
        df_test = self.test.df.copy()
        df_train["split"] = "train"
        df_test["split"] = "test"
        self.df_all = pd.concat([df_train, df_test], axis=0)


    def plot_pairplot(self):
        g = sns.pairplot(
            self.df_all,
            hue="split",
            diag_kind="kde",
            plot_kws={
                "alpha": 0.6,
                "s": 25,
                "edgecolor": "white",
                "linewidth": 0.5,
            },
            diag_kws={
                "fill": True,
                "linewidth": 1.0,
            }
        )
    
        # Apply grid to all subplots
        for ax in g.axes.flatten():
            if ax is not None:
                ax.grid(True, linestyle="--", alpha=0.6)
    
        # --- Remove duplicate legends from inner axes ---
        for ax in g.axes.flatten():
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()
    
        # --- Add a single clean legend ---
        g.fig.legend(
            labels=g._legend_data.keys(),
            handles=g._legend_data.values(),
            loc="upper right",
            bbox_to_anchor=(1.02, 0.98),  # move slightly outside
            title="Split"
        )
    
        # Figure title
        g.fig.suptitle("Pairplot of Train vs. Test", fontsize=16, weight="bold")
        g.fig.tight_layout()
        g.fig.subplots_adjust(top=0.95)
    
        plt.show()


    def plot_distribution(self, feat):
        sns.histplot(
            data = self.df_all,
            x = feat,
            bins = 50,
            stat = "density",
            fill = False,
            hue = "split",
            kde = True,     # also add KDE curves
            alpha = 0.5
        )
        plt.grid(True, linestyle = "--", alpha = 0.6)
        plt.title(f"Distribution of {feat} of Train vs. Test", fontsize = 14, weight = "bold")
        plt.tight_layout()
        plt.show()


    def print_stat_sum(self):
        print("==" * 20 + "Train data Statistics Summary" + "==" * 20)
        self.train.print_stat_sum()
        print("==" * 20 + "Test data Statistics Summary" + "==" * 20)
        self.test.print_stat_sum()


    def plot_boxplot(self):
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # ------- Train -------
        df_scaled_train = pd.DataFrame(
            StandardScaler().fit_transform(self.train.df),
            columns=self.train.df.columns
        )
        
        sns.boxplot(data=df_scaled_train, orient="h", ax=axes[0])
        axes[0].set_xlabel("Standardized Value (z-score)")
        axes[0].set_title("Train: Standardized Features", fontsize=14, weight="bold")
        axes[0].grid(True, linestyle="--", alpha=0.6)
        
        # ------- Test -------
        df_scaled_test = pd.DataFrame(
            StandardScaler().fit_transform(self.test.df),
            columns=self.test.df.columns
        )
        
        sns.boxplot(data=df_scaled_test, orient="h", ax=axes[1])
        axes[1].set_xlabel("Standardized Value (z-score)")
        axes[1].set_title("Test: Standardized Features", fontsize=14, weight="bold")
        axes[1].grid(True, linestyle="--", alpha=0.6)
        
        plt.tight_layout()
        plt.show()


    def plot_corr_heatmap(self):
        print("==" * 20 + "Train data Correlation Heatmap" + "==" * 20)
        self.train.plot_corr_heatmap()
        print("==" * 20 + "Test data Correlation Heatmap" + "==" * 20)
        self.test.plot_corr_heatmap()
                        