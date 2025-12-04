# Out-of-distribution Data Fitting

Tools, datasets, and experiments for studying **out-of-distribution (OOD) generalization** in tabular data, with a focus on realistic regression tasks.

---

## 🔍 Overview

This repository contains code and notebooks for:

- Building and analysing **tabular benchmarks** for OOD data fitting  
- Exploring different **partition algorithms** for creating OOD splits  
- Performing **exploratory data analysis (EDA)** and statistical summaries  
- Prototyping **ML models** and evaluating them under distribution shift

Most examples currently use the **California housing** data stored under `data/`, but the code is structured so you can plug in other tabular datasets.

---

## 📁 Repository Structure

Update this section to match your repo exactly.

- `data/`  
  - Raw and/or preprocessed datasets (e.g. California housing parquet files)
- `partition_algs/`  
  - Scripts / utilities for creating in-distribution vs OOD partitions
- `ML_techniques_list.xlsx`  
  - Notes or taxonomy of ML techniques used / considered
- `*.ipynb`  
  - Jupyter notebooks for experiments, EDA, and visualisation
- `README.md`  
  - Project description and usage (this file)

If you have additional folders (e.g. `src/`, `notebooks/`, `results/`), describe them here as well.

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

If you have a `requirements.txt`:

    pip install -r requirements.txt

Or document your preferred environment setup here (e.g. `conda env create -f environment.yml`).

---

## 📊 Data

Describe your main datasets here. For example:

- **California housing dataset**
  - Stored as a parquet or CSV file in `data/` (e.g. `data/raw/california/california.parquet`)
  - Columns may include: longitude, latitude, median income, house age, etc.
  - Target: e.g. median house value (update this with the real target name)

If some data files are too large for GitHub, explain how to download or generate them (add links or scripts if available).

---

## 🧪 Usage Examples

Adjust imports and paths to match your actual code structure.

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

Explain how to use the code in `partition_algs/`. For example:

    # Example placeholder – replace with real usage
    from partition_algs.some_module import create_partitions

    splits = create_partitions(df, target_column="median_house_value")
    train_id, test_ood = splits["train_id"], splits["test_ood"]

Add more examples for:
- Creating OOD splits
- Training a model on in-distribution data
- Evaluating performance on OOD splits

---

## 📈 Experiments & Notebooks

Use this section to list key notebooks and what they do, for example:

- `01_eda_california.ipynb` – basic EDA and visualisation  
- `02_partition_comparison.ipynb` – compare different OOD splitting strategies  
- `03_model_baselines.ipynb` – baseline models and metrics under OOD settings  

For each notebook, briefly describe:
- Input data
- Main goals
- Key outputs (plots, metrics, tables)

---

## ✅ Roadmap / TODOs

You can keep a simple checklist here:

- [ ] Add more tabular datasets (e.g. gas sensors, taxi, etc.)  
- [ ] Implement additional partition algorithms  
- [ ] Add baseline models (e.g. Linear Regression, Random Forest, XGBoost, CatBoost)  
- [ ] Implement standard OOD metrics and evaluation scripts  
- [ ] Write unit tests for core utilities

---

## 🤝 Contributing

If this is just for personal research, you can keep this simple.  
If you plan to accept contributions:

1. Fork the repo  
2. Create a feature branch: `git checkout -b feature/my-feature`  
3. Commit your changes: `git commit -m "Add my feature"`  
4. Push to the branch: `git push origin feature/my-feature`  
5. Open a pull request

Please try to keep notebooks clean (remove large outputs) and document new scripts.

---

## 📄 License

Specify your license here, for example:

This project is licensed under the **MIT License** – see the `LICENSE` file for details.

---

## 📚 References

Add papers, datasets, and libraries you rely on. For example:

- Papers on out-of-distribution generalisation
- Original sources of the datasets
- Key libraries: pandas, NumPy, scikit-learn, matplotlib, seaborn, etc.

---

## ✉️ Contact

Maintainer: `<Your Name>`  
GitHub: [@Trminh06-work](https://github.com/Trminh06-work)

Feel free to open an issue if you have questions, suggestions, or find a bug.
