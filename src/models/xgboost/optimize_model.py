import json

import optuna
import numpy as np
from sklearn import metrics

from src.utils import read_data
from train_model import get_model_class


def objective(trial, train, val, n_jobs, early_stopping_rounds):
    params = {
        "verbosity": 0,
        "objective": "reg:squarederror",
        "n_estimators": 10000,
        "max_depth": trial.suggest_int("max_depth", 4, 30),
        "min_child_weight": trial.suggest_loguniform("min_child_weight", 1, 1000),
        "colsample_bytree": trial.suggest_loguniform("colsample_bytree", 0.05, 0.6),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.005, 0.05),
        "subsample": trial.suggest_loguniform("subsample", 0.4, 1.0),
        "alpha": trial.suggest_loguniform("alpha", 0.01, 10.0),
        "lambda": trial.suggest_loguniform("lambda", 1e-8, 10.0),
        "gamma": trial.suggest_loguniform("lambda", 1e-8, 10.0),
        "n_jobs": n_jobs,
    }

    model = get_model_class()(**params)
    pruning_callback = optuna.integration.XGBoostPruningCallback(
        trial, "validation_0-rmse"
    )

    model.fit(
        train["x"],
        np.log10(train["y"] + 1),
        eval_set=[(val["x"], np.log10(val["y"] + 1))],
        eval_metric="rmse",
        callbacks=[pruning_callback],
        early_stopping_rounds=early_stopping_rounds,
        verbose=0,
    )

    preds = 10**model.predict(val['x']) - 1
    mean_absolute_error = metrics.mean_absolute_error(val['y'], preds)
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
            early_stopping_rounds=50,
        ),
        n_trials=1000,
        timeout=2400,
        show_progress_bar=True,
    )

    trial = study.best_trial
    with open("src/models/xgboost/best_params.json", "w", encoding="utf8") as fp:
        json.dump(trial.params, fp)
