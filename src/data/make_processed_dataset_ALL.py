import pandas as pd
from sklearn import model_selection

from src.utils import seed_everything

SEED = 357

def train_test_split_by_income(data, test_size=0.2, bins=5):
    income_bined = pd.qcut(data['Y'], q=bins)

    return model_selection.train_test_split(data, test_size=test_size, shuffle=True, stratify=income_bined, random_state=SEED)


if __name__ == '__main__':
    seed_everything(SEED)

    """
    napisane, żeby było, a nie wyglądało, więc niezbyt czytelne
    """
    def get_mapping(l1, l2):
        mapping = {e1: e2 for e1, e2 in zip(l1, l2)}

        return mapping

    data_old: pd.DataFrame = pd.read_csv('data/raw/temat_2_dane.csv')
    data_old_desc = pd.read_excel('data/raw/temat_2_opis_zmiennych.xlsx')
    data_old_desc['OPIS'] = [" ".join(e.split()).lower() for e in data_old_desc['OPIS'].values]
    data_old_desc_mapping = get_mapping(data_old_desc['NAZWA'].values, data_old_desc['OPIS'].values)
    data_old = data_old.rename(data_old_desc_mapping, axis=1)

    data_new: pd.DataFrame = pd.read_csv('data/raw/temat_2_dane_NEW.csv', index_col=0)
    data_new_desc = pd.read_excel('data/raw/temat_2_opis_zmiennych_NEW.xlsx')
    data_new_desc['OPIS'] = [" ".join(e.split()).lower() for e in data_new_desc['OPIS'].values]
    data_new_desc_mapping = get_mapping(data_new_desc['NAZWA'].values, data_new_desc['OPIS'].values)
    data_new = data_new.rename(data_new_desc_mapping, axis=1)

    common_desc = (data_new_desc[data_new_desc['OPIS'].isin(data_old_desc['OPIS'].values)]['OPIS'].values)
    inverse_mapping = {str(common_desc[0]): 'Y'}
    for i, desc in enumerate(common_desc[1:], start=1):
        inverse_mapping[desc] = f'X{i}'

    data_common = pd.concat([data_old[common_desc], data_new[common_desc]], ignore_index=True).rename(inverse_mapping, axis=1)

    data_train_val, data_test = train_test_split_by_income(data_common)
    # test_size=0.25 instead of 0.2 to keep 60/20/20-train/val/test split
    data_train, data_val = train_test_split_by_income(data_train_val, test_size=0.25)

    data_train.to_csv('data/processed/data_train_ALL.csv', index=False)
    data_val.to_csv('data/processed/data_val_ALL.csv', index=False)
    data_test.to_csv('data/processed/data_test_ALL.csv', index=False)

    pd.DataFrame.from_dict({'ZMIENNA': list(inverse_mapping.values()), 'OPIS': list(inverse_mapping.keys())})\
        .to_csv('data/processed/data_ALL_opis_zmiennych.csv', index=False, sep=';')
