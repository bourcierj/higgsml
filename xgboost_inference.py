"""Inference on Higgs ML Data with XGBOOST"""

import os
import argparse
import pickle

import matplotlib.pyplot as plt
plt.style.use('ggplot')
import xgboost as xgb
import numpy as np
from hyperopt import space_eval

from xgboost_hyperopt import PARAMS_SPACE
from utils import *
from plot_utils import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, help='random generators seed (default: None)')
    parser.add_argument('--logdir', type=str, default='./trials', help='save directory')
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    xgb_seed = 0
    if args.seed is not None:
        # random seed for reproducibility
        np.random.seed(42)
        xgb_seed = args.seed
    PARAMS_SPACE['seed'] = xgb_seed

    df_train, df_lead, df_test, _, = get_cern_datasets()

    # prepare datasets
    train_data, train_target, train_weights = prepare_cern_dataset(df_train)
    test_data, test_target, test_weights = prepare_cern_dataset(df_test)
    lead_data, lead_target, lead_weights = prepare_cern_dataset(df_lead)

    # xgboost DMatrices
    dtrain = xgb.DMatrix(train_data, label=train_target)
    dtest = xgb.DMatrix(test_data, label=test_target)
    dlead = xgb.DMatrix(lead_data, label=lead_target)

    # load Trials object from hyperopt search in log dir
    with open(os.path.join(args.logdir, 'Trials-xgb.pkl'), 'rb') as file:
        trials = pickle.load(file)

    #print("All Losses:\n", trials.losses())
    #print("First result\n:", trials.results[0])
    # get all losses
    all_ams = [-ams for ams in trials.losses()]
    # extract the best Trial among all
    bidx = np.argmax(all_ams)
    bresult = trials.results[bidx]
    ams, ams_var = -bresult['loss'], bresult['loss_variance'],
    threshold, threshold_var = bresult['threshold'], bresult['threshold_variance']
    xgb_ntree, xgb_ntree_var = bresult['xgb_ntree'], bresult['xgb_ntree_variance']

    print('Hyperopt best results:\n'
          '\tMean AMS: {:.6f}\n'
          '\tVar AMS: {:.6f}\n'
          '\tMean threshold: {:.6f}\n'
          '\tVar threshold: {:.6f}\n'
          '\tMean XGB ntree: {:.6f}\n'
          '\tVar XGB ntree: {:.6f}\n'
          '\tTrial: {}\n'
          .format(ams, ams_var, threshold, threshold_var, xgb_ntree, xgb_ntree_var, bidx))

    # # plot ams with trials
    # fig, ax = plt.subplots(figsize=(9,6))
    # plot_ams_with_trials(trials, ax=ax)
    # plt.savefig('figures/xgboost/AMS Score with Trials.svg')
    # plt.show()

    # # plot ams curves for each fold
    # fig, ax = plt.subplots(figsize=(9,6))
    # plot_cv_ams_curves_for_trial(bresult['cv_ams_curves'], ax=ax)
    # plt.savefig('figures/xgboost/CV AMS Curves.svg')
    # plt.show()

    bparams = trials.argmin
    bparams = space_eval(PARAMS_SPACE, bparams)
    # retrain the model with best set of parameters on full training set
    num_round = int(bparams.pop('num_boost_round'))
    print('Training on train set...')
    gbm = xgb.train(bparams, dtrain, num_round)
    print('Done\n')

    # predictions with best threshold
    print("Scores with best threshold in CV:")
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


# Hyperopt best results:

#     Mean AMS: 3.601115
#     Var AMS: 0.002257
#     Mean threshold: 0.854108
#     Var threshold: 0.002407
#     Mean XGB ntree: 57.600000
#     Var XGB ntree: 169.040000
#     Trial: 92

# RESULTS, WITH BEST CV THRESHOLD

# Train AMS: 4.175217
# Test AMS: 3.490362
# Leaderboard AMS: 3.388939

# Note: the num_round hyperparam is the value found in the param space, it is NOT set to
# mean xgb ntree. The score is better:

# WITH NUM_ROUND SET AS CEIL(MEAN XGB_NTREE): Test score drops of 0.049, very close

# Train AMS: 4.020941
# Test AMS: 3.487744
# Leaderboard AMS: 3.361855

# RESULTS, WITH BEST THRESHOLD ON TRAIN: Big overfitting

# Best threshold on train: 0.919840
# Best threshold on test: 0.843687

# Train AMS: 4.206838
# Test AMS: 3.085809
# Leaderboard AMS: 2.893223
