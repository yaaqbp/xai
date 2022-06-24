import pandas as pd
import xgboost
import numpy as np

from src.data.make_processed_dataset_NEW import train_test_split_by_income
from src.utils import seed_everything, read_data

if __name__ == "__main__":
    # seedujemy
    SEED = seed_everything(357)

    # parametry xgboosta
    params_regressor = {
        "max_depth": 37,
        "min_child_weight": 4.298451835693228,
        "colsample_bytree": 0.5043465069716095,
        "learning_rate": 0.008597619707703471,
        "subsample": 0.809749228171027,
        "alpha": 0.14057268517512703,
        "lambda": 1.0031006429605082e-06,
        "random_state": SEED,
    }



    # ustaliliśmy parametry przy pomocy zbioru walidacyjnego, to teraz by wytrenować ostateczny model
    # łączymy zbiór walidacyjny i treningowy, a następnie wydzielamy 10% z tego połączenia, jako dane
    # do wczesnego zatrzymania ('early stopping')
    X_train, y_train = read_data("data/processed/data_train_NEW.csv")
    X_val, y_val = read_data("data/processed/data_val_NEW.csv")

    X_train = pd.concat([X_train, X_val])
    y_train = pd.concat([y_train, y_val])

    data_train = X_train
    data_train["Y"] = y_train

    data_train, data_val = train_test_split_by_income(data_train, test_size=0.10)

    X_train = data_train.drop("Y", axis=1)
    y_train = data_train["Y"]

    X_es = data_val.drop("Y", axis=1)
    y_es = data_val["Y"]



    model = xgboost.XGBRegressor(
        objective="reg:squarederror", n_estimators=10000, **params_regressor
    )

    model.fit(
        X_train,
        np.log10(y_train + 1),
        eval_set=[(X_es, np.log10(y_es + 1))],
        eval_metric="rmse",
        early_stopping_rounds=50,
    )

    model.save_model("models/xgboost.json")



