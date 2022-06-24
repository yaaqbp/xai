import json

from sklearn import ensemble
import joblib

from src.utils import read_data, seed_everything


def get_model_class():
    return ensemble.RandomForestRegressor

if __name__ == '__main__':
    seed_everything(357)

    with open('src/models/randomforest/best_params.json') as fp:
        params = json.load(fp)
    model = get_model_class()(**params)

    X_train, y_train = read_data('data/processed/data_train.csv')
    X_val, y_val = read_data('data/processed/data_val.csv')

    model.fit(X_train, y_train)
    joblib.dump(model, 'models/randomforest.json')
