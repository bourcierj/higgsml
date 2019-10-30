"""XGBOOST training with default parameters. No hyperparameters search here."""

import argparse

import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import xgboost as xgb

from utils import *
from plot_utils import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, help='random generators seed (default: None)')
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    xgb_seed = 0
    if args.seed is not None:
        # random seed for reproducibility
        np.random.seed(42)
        xgb_seed = args.seed

    df_train, df_lead, df_test, _, = get_cern_datasets()

    # prepare datasets
    train_data, train_target, train_weights = prepare_cern_dataset(df_train)
    test_data, test_target, test_weights = prepare_cern_dataset(df_test)
    lead_data, lead_target, lead_weights = prepare_cern_dataset(df_lead)

    # xgboost DMatrices
    dtrain = xgb.DMatrix(train_data, label=train_target)
    dtest = xgb.DMatrix(test_data, label=test_target)
    dlead = xgb.DMatrix(lead_data, label=lead_target)

    params = {
        'objective': 'binary:logistic',
        'metric': 'auc',
        'seed': xgb_seed
    }
    print('Training on train set...')
    gbm = xgb.train(params, dtrain)
    # predictions with default threshold of 0.5
    threshold = 0.854108
    train_preds = gbm.predict(dtrain)
    train_ams = ams_score(train_target, round_predictions(train_preds, threshold),
                          train_weights)

    print(f'Train AMS: {train_ams:.6f}')

    test_preds = gbm.predict(dtest)
    test_ams = ams_score(test_target, round_predictions(test_preds, threshold),
                         test_weights)
    print(f'Test AMS: {test_ams:.6f}')

    lead_preds = gbm.predict(dlead)
    lead_ams = ams_score(lead_target, round_predictions(lead_preds, threshold),
                         lead_weights)
    print(f'Leaderboard AMS: {lead_ams:.6f}')


# RESULTS WITH DEFAULT HPARAMS AND DEFAULT THRESHOLD OF 0.5:

# Train AMS: 2.354525
# Test AMS: 2.355997
# Leaderboard AMS: 2.313867


# RESULTS WITH DEFAULT HPARAMS AND OPTIMAL THRESHOLD OF 0.8541:

