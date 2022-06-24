import json

import optuna
from sklearn import metrics

from src.utils import read_data
from train_model import get_model_class


def objective(trial, train, val, n_jobs):
    params = {
        "n_estimators": trial.suggest_int('n_estimators', 100, 1000),
        "max_features": trial.suggest_loguniform('max_features', 0.05, 1.0),
        "n_jobs": n_jobs,
    }

    model = get_model_class()(**params)

    model.fit(
        train["x"],
        train["y"],
    )

    preds = model.predict(X_val)
    mean_absolute_error = metrics.mean_absolute_error(y_val, preds)
    return mean_absolute_error


if __name__ == "__main__":
    X_train, y_train = read_data("data/processed/data_train.csv")
    X_val, y_val = read_data("data/processed/data_val.csv")

    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(
            trial,
            train={"x": X_train, "y": y_train},
            val={"x": X_val, "y": y_val},
            n_jobs=4,
        ),
        n_trials=1000,
        timeout=60 * 10,
        show_progress_bar=True,
    )

    trial = study.best_trial
    with open("src/models/randomforest/best_params.json", "w", encoding="utf8") as fp:
        json.dump(trial.params, fp)
