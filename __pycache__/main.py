import pandas as pd
import pickle

from sklearn.model_selection import StratifiedKFold

import model_lightgbm
import model_xgboost
import numpy as np
import xgboost as xgb


def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f), pickle.load(f)

if __name__ == '__main__':
    import data_process
    import model_xgboost
    import model_lightgbm

    xgb_train, xgb_test = load_pickle('xgboost_result.pkl')
    lgb_train, lgb_test = load_pickle('lightgbm_result.pkl')

    meta_train = np.hstack((xgb_train, lgb_train))
    meta_test = np.hstack((xgb_test, lgb_test))

    meta_train_labels = pd.read_csv('security_train.csv')['label'].values

    skf = StratifiedKFold(n_splits=5, random_state=4, shuffle=True)
    final_train = np.zeros((meta_train.shape[0], 8))
    final_test = np.zeros((meta_test.shape[0], 8))

    for train_idx, val_idx in skf.split(meta_train, meta_train_labels):
        train_data, train_labels = meta_train[train_idx], meta_train_labels[train_idx]
        val_data, val_labels = meta_train[val_idx], meta_train_labels[val_idx]

        dtrain = xgb.DMatrix(train_data, label=train_labels)
        dval = xgb.DMatrix(val_data, label=val_labels)
        dtest = xgb.DMatrix(meta_test)

        param = {'max_depth': 6, 'eta': 0.1, 'eval_metric': 'mlogloss', 'silent': 1, 'objective': 'multi:softprob',
                 'num_class': 8, 'subsample': 0.8, 'colsample_bytree': 0.85}

        evallist = [(dtrain, 'train'), (dval, 'val')]
        num_round = 300
        final_model = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=50)

        final_train[val_idx] = final_model.predict(dval)
        final_test += final_model.predict(dtest) / 5.0

    test_submit = pd.read_csv('security_test.csv')[['file_id']]
    test_submit[['prob0', 'prob1', 'prob2', 'prob3', 'prob4', 'prob5', 'prob6', 'prob7']] = final_test
    test_submit.to_csv('final_submission.csv', index=False)
