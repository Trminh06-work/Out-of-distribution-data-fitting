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
│   ├── splitted/                         # Partitioned datasets for evaluation
│   ├── stat_sum_func.py                  # Script / utilities to perform basic EDA and data visualization
│   └── Data_Statistics_Summary.ipynb     # Notebook to perform basic EDA and data visualization on raw/*
│
├── partition_algs/
│   ├── random_split.py                   # Script for Meta-feature based partitioning technique, representative of Concept Shifts
│   ├── mfs_split.py                      # Script for Meta-feature based partitioning technique, representative of Concept Shifts
│   ├── modified_mfs_split.py             # A modified version of mfs_split.py to suit the need of this project
│   ├── marginal_distribution_split.py    # Script for Distribution based partitioning techniques, i.e. Covariate shift and Prior shift
│   ├── geometric_split.py                # Script for Geometry based partitioning techniques, i.e. Hyperballs and Slabs
│   └── PartitionProcedure.ipynb          # Notebook to split the datasets using the above approaches
│
│
├── models/
│   ├── Results/                          # Main results (baseline experiment)
│   ├── Results_add/                      # Extra results (side experiment)
│   ├── EvaluationToolbox.py              # Script to score models' performance, sketch Partial Dependence Plots, and hyparameters tunning engine
│   ├── Experiment.py                     # Script for csv/tsv -> parquet conversion, evaluate models across datasets and split regimes, statistical analysis tools
│   ├── Experiments.ipynb                 # Notebook to evaluate models to record results
│   ├── HypoTest.ipynb                    # Notebook to perform hypotheses tests, e.g. best model, performance table, etc...
│   ├── Models.py                         # Script to add models being evaluated
│   ├── ModelsBenchmark.ipynb             # Notebook to verify models before officially launching experiment using all datasets.
│   └── ft_transformer.py                 # Script for FT-Transformer, adapted from external sources
│
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

## 🧪 Data uploading and Partitioning

### Upload csv/tsv files to raw/ folder

### Load data and compute basic statistics
    data/Data_Statistics_Summary.ipynb

### Run partition algorithms

    parition_algs/PartitionProcedure.ipynb

Due to GitHub resources constraints, the splitted data are not pushed to this repo. Hence, practitioners must re-run the `PartitionProcedure.ipynb` to obtain the splitted data. However, the statistics summary of the splitted data is provided to compare.

### Load splitted data and compute basic statistics

    data/Splitted_Data_Statistics_Summary.ipynb
---

## 📈 Experiments & Notebooks

1. Models Verification before experiment -> use `models/ModelsBenchmark.ipynb`
2. Main and Side experiment -> use `models/Experiment.ipynb`
3. Analysis -> use `models/HypoTest.ipynb`

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
