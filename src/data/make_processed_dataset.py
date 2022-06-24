import pandas as pd
from sklearn import model_selection

from src.utils import seed_everything

SEED = 357

def train_test_split_by_income(data, test_size=0.2, bins=5):
    income_bined = pd.qcut(data['Y'], q=bins)

    return model_selection.train_test_split(data, test_size=test_size, shuffle=True, stratify=income_bined, random_state=SEED)


if __name__ == '__main__':
    seed_everything(SEED)

    data: pd.DataFrame = pd.read_csv('data/raw/temat_2_dane.csv')

    data_train_val, data_test = train_test_split_by_income(data)
    # test_size=0.25 instead of 0.2 to keep 60/20/20-train/val/test split
    data_train, data_val = train_test_split_by_income(data_train_val, test_size=0.25)

    data_train.to_csv('data/processed/data_train.csv', index=False)
    data_val.to_csv('data/processed/data_val.csv', index=False)
    data_test.to_csv('data/processed/data_test.csv', index=False)
