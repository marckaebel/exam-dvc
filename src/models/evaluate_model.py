import pandas as pd
import numpy as np
from joblib import load
import json
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score

X_train = pd.read_csv('data/processed/X_train_scaled.csv')
X_test = pd.read_csv('data/processed/X_test_scaled.csv')
y_train = pd.read_csv('data/processed/y_train.csv')
y_test = pd.read_csv('data/processed/y_test.csv')
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)


def main(repo_path):
    model = load(repo_path / "models/trained_model.joblib")
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    metrics = {"mse": mse, "r2": r2}
    accuracy_path = repo_path / "metrics/scores.json"
    accuracy_path.write_text(json.dumps(metrics))
    # Save predictions
    predictions_path = repo_path / "data/predictions.csv"
    pd.DataFrame({
        "prediction": predictions
    }).to_csv(predictions_path, index=False)


if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent.parent
    main(repo_path)
