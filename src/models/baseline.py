import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

def majority_class_baseline(y_train, y_test):
    """
    Predict the most common class in y_train for every test item.
    Returns (accuracy, majority_label).
    """
    majority_label = int(pd.Series(y_train).mode()[0])
    y_pred = np.full_like(y_test, fill_value=majority_label)
    acc = accuracy_score(y_test, y_pred)
    return acc, majority_label

def logistic_regression_pipeline(df_train, df_test, y_train, y_test):
    """
    Logistic regression baseline:
    - One-hot encode categorical columns
    - Standardize numeric columns
    - Fit LogisticRegression
    - Return accuracy and a classification report
    """
    categorical_cols = df_train.select_dtypes(include="object").columns.tolist()
    numeric_cols = df_train.select_dtypes(exclude="object").columns.tolist()

    # If the string target 'income' column is still present, drop it
    if "income" in df_train.columns:
        df_train = df_train.drop(columns=["income"])
        df_test = df_test.drop(columns=["income"])
        if "income" in categorical_cols:
            categorical_cols.remove("income")

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", StandardScaler(), numeric_cols),
        ]
    )

    model = LogisticRegression(max_iter=1000)

    clf = Pipeline(steps=[
        ("pre", preprocessor),
        ("logreg", model)
    ])

    clf.fit(df_train, y_train)
    y_pred = clf.predict(df_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=False)

    return acc, report
