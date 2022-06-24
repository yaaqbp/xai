import json

import xgboost as xgb
import numpy as np
from sklearn import metrics, pipeline

from src.utils import read_data, seed_everything


def get_model_class():
    return xgb.XGBRegressor

if __name__ == '__main__':
    seed_everything(357)

    with open('src/models/xgboost/best_params.json') as fp:
        params = json.load(fp)
    model = get_model_class()(verbosity=0, objective='reg:squarederror', n_estimators=10000, **params)

    X_train, y_train = read_data('data/processed/data_train.csv')
    X_val, y_val = read_data('data/processed/data_val.csv')

    model.fit(X_train, np.log10(y_train + 1), eval_set=[(X_val, np.log10(y_val + 1))], eval_metric='mae', early_stopping_rounds=50)
    print(metrics.mean_absolute_error(y_val, 10**model.predict(X_val) - 1))

    model.save_model('models/xgboost.json')
