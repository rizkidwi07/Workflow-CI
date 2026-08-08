import argparse
import os

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


def main(args):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    data_path = args.data_path
    if not os.path.isabs(data_path):
        data_path = os.path.join(BASE_DIR, data_path)

    # Muat dataset hasil preprocessing
    data = pd.read_csv(data_path)
    X = data.drop("Churn", axis=1)
    y = data["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Autolog: parameter, metrik, dan model dicatat otomatis
    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="RandomForest_CI"):
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            random_state=42,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_f1_score", f1)

        print("Test accuracy:", acc)
        print("Test f1-score:", f1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="telco_churn_preprocessed.csv")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--min_samples_split", type=int, default=5)
    args = parser.parse_args()
    main(args)
