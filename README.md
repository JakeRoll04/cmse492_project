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

## Setup Instructions
1. Create and activate a virtual environment:
```bash
python -m venv venv
# macOS / Linux:
source venv/bin/activate
# Windows PowerShell:
venv\Scripts\Activate.ps1