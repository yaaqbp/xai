#!/usr/bin/env python
# coding: utf-8
# %%
import pandas as pd
import numpy as np

def log(df, feature_list=None):
    sort = df.columns
    if feature_list == None:
        feature_list = sort
    fdf = df[feature_list]
    fdf = np.log10(fdf+1)
    df = df[set(list(set(df.columns)-set(feature_list)))]
    return pd.concat([df, fdf], axis = 1).reindex(sort, axis = 1)

if __name__ == '__main__':
    df_train = pd.read_csv('../../data/processed/data_train.csv')
    df_val = pd.read_csv('../../data/processed/data_val.csv')
    df_test = pd.read_csv('../../data/processed/data_test.csv')

    with open('money_features.txt') as f:
        money_features = f.read().splitlines()
    money_features.append('Y')

    df_train[df_train.Y > 20000] = 20000

    log(df_train, money_features).to_csv('../../data/processed/log_trans/data_train.csv', index = False)
    log(df_test, money_features).to_csv('../../data/processed/log_trans/data_test.csv', index = False)
    log(df_val, money_features).to_csv('../../data/processed/log_trans/data_val.csv', index = False)

