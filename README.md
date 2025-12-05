# Out-of-distribution Data Fitting

A benchmark for studying **out-of-distribution (OOD) generalization** of ML/DL models on low-dimensional (up to 20 features), regression tabular data.

---

## 🎯 Key contributions

This repository encompasses code and documents for:

- Building and analysing **regression tabular benchmarks** for OOD data fitting  
- Exploring different **partition algorithms** for creating OOD partitions
- Performing **exploratory data analysis (EDA)** and **data visualization**
- Prototyping **ML/DL models** and evaluating them under distributional shifts and OOD extrapolation.

---

## 📁 Repository Structure

```text
├── Report/
│   ├── project description/              # Relevant documents used and brief description about this project
│   └── Report.pdf                        # The main documentation of this project
│
├── data/                     
│   ├── raw/                              # Cleaned datasets
│   └── splitted/                         # Partitioned datasets for evaluation
│   └── stat_sum_func.py                  # Script / utilities to perform basic EDA and data visualization
│   └── Data_Statistics_Summary.ipynb     # Notebook to perform basic EDA and data visualization on raw/*
│
├── partition_algs/
│   ├── mfs_split.py                      # Script for Meta-feature based partitioning technique
│   ├── marginal_distribution_split.py    # Script for Distribution based partitioning techniques, i.e. Covariate/Prior shift and Concept shift
│   ├── geometric_split.py                # Script for Geometric partioning techniques, i.e. Hyperballs and Slabs
│   └── PartitionProcedure.ipynb          # Notebook to split the datasets using the above approaches
│
├── LICENSE                               # MIT LICENSE
├── README.md                             # Project overview and instructions
└── requirements.txt                      # Dependencies
```

---

## ▶️ Getting Started

### 1. Clone the repository

    git clone https://github.com/Trminh06-work/Out-of-distribution-data-fitting.git
    cd Out-of-distribution-data-fitting

### 2. Create and activate a virtual environment (optional but recommended)

On macOS / Linux:

    python -m venv venv
    source venv/bin/activate

On Windows:

    python -m venv venv
    venv\Scripts\activate

### 3. Install dependencies

    pip install -r requirements.txt

---

## 🧪 Usage Examples

**NOTES (delete later)** Adjust imports and paths to match your actual code structure. 

### Load data and compute basic statistics

    from stat_sum_func import DatasetStatistics  # update path if different

    path = "data/raw/california/california.parquet"
    stats = DatasetStatistics(path)

    # Access underlying dataframe
    df = stats.df
    print(df.head())

    # Plot distribution of a single feature
    stats.plot_distribution("Longitude")

### Run partition algorithms

**NOTES (delete later)** Explain how to use the code in `partition_algs/`. For example:

    # Example placeholder – replace with real usage
    from partition_algs.some_module import create_partitions

    splits = create_partitions(df, target_column="median_house_value")
    train_id, test_ood = splits["train_id"], splits["test_ood"]

---

## 📈 Experiments & Notebooks

**A TEMPLATE FOR THIS SECTION**

Use this section to list key notebooks and what they do, for example:

- `01_eda_california.ipynb` – basic EDA and visualisation  
- `02_partition_comparison.ipynb` – compare different OOD splitting strategies  
- `03_model_baselines.ipynb` – baseline models and metrics under OOD settings  

For each notebook, briefly describe:
- Input data
- Main goals
- Key outputs (plots, metrics, tables)

---

## 📄 License

This project is licensed under the **MIT License** – see the `LICENSE` file for details.

---

## 🧾 Report

Refer to **Report.pdf** for further information about References and the outcomes of this project.

---

## ✉️ Contact

Author: `Bao Minh Tran`  

GitHub: [@Trminh06-work](https://github.com/Trminh06-work)

LinkedIn: [Bao Minh Tran](www.linkedin.com/in/bao-minh-tran-587272372)

Feel free to open an issue if you have questions, suggestions, or find a bug.
