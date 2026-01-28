# Jake Roll - rolljake@msu.edu

# Income Classification from U.S. Census Data

## Project Description
This project predicts whether an individual earns `>50K` per year based on U.S. Census demographic and employment features (age, workclass, education, marital status, occupation, hours worked, etc.). The prediction target is binary (`<=50K` vs `>50K`). We will build baseline and progressively more complex machine learning models and evaluate them for accuracy, fairness, and interpretability.

## Repository Structure

```text
cmse492_project/
├── data/
│   ├── raw/                # original data files (not versioned if large)
│   └── processed/          # cleaned / sampled / model-ready subsets
├── notebooks/
│   └── exploratory/        # exploratory notebooks, EDA, baseline models
├── src/
│   ├── preprocessing/      # preprocessing and splitting scripts
│   ├── models/             # model definitions (baseline, etc.)
│   └── evaluation/         # metrics, reporting utilities
├── figures/                # EDA plots, class balance, gantt chart, etc.
├── docs/                   # notes, references
└── reports/                # LaTeX project proposal and final report
```

---

## Setup Instructions

### Clone the repository
```bash
git clone https://github.com/JakeRoll04/cmse492_project.git
cd cmse492_project
```

---

## Environment Setup (uv)

This project uses **uv** for Python dependency and environment management.

### Prerequisites
- Python 3.11
- uv (install with `pip install uv`)

### Setup
```bash
uv sync
uv run python main.py
```

(Optional) Activate the environment:
```bash
source .venv/bin/activate
```

---

## Data

This project uses the **Adult (Census Income) dataset** derived from the U.S. Census Bureau and distributed by the UCI Machine Learning Repository:

https://archive.ics.uci.edu/dataset/2/adult

### Data Locations
- **Raw data:** `data/raw/adult.data`  
  (Downloaded manually after cloning; not tracked in git)

- **Processed data:** `data/processed/adult_sample.csv`  
  (Generated during reproduction)

### Download Raw Data
```bash
curl -L -o data/raw/adult.data https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
```

---

## Reproduce a Result

The following command loads the raw data, cleans it, saves a processed sample, and reproduces a class-balance figure.

```bash
uv run python -c "
from pathlib import Path
from src.preprocessing.preprocess import load_raw_data, clean_data
import matplotlib.pyplot as plt

DATA_PATH = Path('data/raw/adult.data')
processed_dir = Path('data/processed')
figures_dir = Path('figures')

processed_dir.mkdir(parents=True, exist_ok=True)
figures_dir.mkdir(parents=True, exist_ok=True)

df_raw = load_raw_data(DATA_PATH)
df_clean = clean_data(df_raw)

print('After cleaning:', df_clean.shape)
print(df_clean.head().to_string(index=False))

sample_path = processed_dir / 'adult_sample.csv'
df_clean.sample(1000, random_state=0).to_csv(sample_path, index=False)
print('Saved sample:', sample_path)

fig, ax = plt.subplots(figsize=(5,4))
df_clean['income_binary'].value_counts().plot(kind='bar', ax=ax)
ax.set_xticklabels(['<=50K',' >50K'])
ax.set_ylabel('Count')
ax.set_title('Income Class Balance')
plt.tight_layout()

out_path = figures_dir / 'class_balance.png'
plt.savefig(out_path, dpi=300)
print('Saved figure:', out_path)
"
```

### Expected Outputs
- `data/processed/adult_sample.csv`
- `figures/class_balance.png`
