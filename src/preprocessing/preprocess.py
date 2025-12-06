import pandas as pd
from sklearn.model_selection import train_test_split

COLUMN_NAMES = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income"
]

def load_raw_data(path: str) -> pd.DataFrame:
    """
    Load the raw Adult Income data (U.S. Census income dataset).
    The original file has no header, so we supply column names.
    We also treat '?' as missing.
    """
    df = pd.read_csv(
        path,
        header=None,
        names=COLUMN_NAMES,
        na_values=["?"],
        skipinitialspace=True
    )
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning steps:
    - Strip whitespace from string columns
    - Drop rows with missing values (for now)
    - Create binary target column: income_binary = 1 if >50K else 0
    """
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    df_clean = df.dropna(axis=0).copy()

    df_clean["income_binary"] = (df_clean["income"] == ">50K").astype(int)

    df_clean = df_clean.drop(columns=["income"])

    df_clean = pd.get_dummies(df_clean, drop_first=True)

    return df_clean

def train_val_test_split(df: pd.DataFrame, test_size=0.2, val_size=0.1, random_state=42):
    """
    Create train / val / test splits with stratification on the binary target.
    1) Split temp vs test
    2) Split temp into train vs val
    """
    X = df.drop(columns=["income_binary"])
    y = df["income_binary"]

    # first split off test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    # now split train vs val from temp
    val_ratio = val_size / (1 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        stratify=y_temp,
        random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
