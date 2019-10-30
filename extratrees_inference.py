"""Inference on Higgs ML Data with random forest"""

import os
import argparse
import pickle

import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
from hyperopt import space_eval
import pandas as pd
from sklearn.impute import SimpleImputer

from extratrees_hyperopt import PARAMS_SPACE
from utils import *
from plot_utils import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, help='random generators seed (default: None)')
    parser.add_argument('--logdir', type=str, default='./', help='save directory')
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()

    if args.seed is not None:
        # random seed for reproducibility
        np.random.seed(42)

    df_train, df_lead, df_test, _, = get_cern_datasets()

    # prepare datasets
    imputer = SimpleImputer(strategy='median')
    train_data, train_target, train_weights = prepare_cern_dataset(df_train, True, imputer)
    test_data, test_target, test_weights = prepare_cern_dataset(df_test, False, imputer)
    lead_data, lead_target, lead_weights = prepare_cern_dataset(df_lead, False, imputer)

    # load Trials object from hyperopt search in log dir
    with open(os.path.join(args.logdir, 'Trials-extratrees.pkl'), 'rb') as file:
        trials = pickle.load(file)

    all_ams = [-ams for ams in trials.losses()]
    # extract the best Trial among all
    bidx = np.argmax(all_ams)
    bresult = trials.results[bidx]
    ams, ams_var = -bresult['loss'], bresult['loss_variance'],
    threshold, threshold_var = bresult['threshold'], bresult['threshold_variance']

    print('Hyperopt best results:\n'
          '\tMean AMS: {:.6f}\n'
          '\tVar AMS: {:.6f}\n'
          '\tMean threshold: {:.6f}\n'
          '\tVar threshold: {:.6f}\n'
          '\tTrial: {}\n'
          .format(ams, ams_var, threshold, threshold_var, bidx))
    #pprint(trials.trials)

    # # plot ams with trials
    # fig, ax = plt.subplots(figsize=(9,6))
    # plot_ams_with_trials(trials, ax=ax)
    # plt.savefig('figures/extratrees/AMS Score with Trials.svg')
    # plt.show()

    # # plot ams curves for each fold
    # fig, ax = plt.subplots(figsize=(9,6))
    # plot_cv_ams_curves_for_trial(bresult['cv_ams_curves'], ax=ax)
    # plt.savefig('figures/extratrees/CV AMS Curves.svg')
    # plt.show()

    bparams = trials.argmin
    bparams = space_eval(PARAMS_SPACE, bparams)
    # retrain the model with best set of parameters on full training set
    print('Training on train set...')
    clf = ExtraTreesClassifier(**bparams)
    clf.fit(train_data, train_target)
    print('Done\n')

    # predictions with best threshold
    print("Scores with best threshold in CV:")
    train_preds = clf.predict_proba(train_data)[:, 1]
    train_ams = ams_score(train_target, round_predictions(train_preds, threshold),
                          train_weights)
    print(f'Train AMS: {train_ams:.6f}')

    test_preds = clf.predict_proba(test_data)[:, 1]
    test_ams = ams_score(test_target, round_predictions(test_preds, threshold),
                         test_weights)
    print(f'Test AMS: {test_ams:.6f}')

    lead_preds = clf.predict_proba(lead_data)[:, 1]
    lead_ams = ams_score(lead_target, round_predictions(lead_preds, threshold),
                         lead_weights)
    print(f'Leaderboard AMS: {lead_ams:.6f}')

## WITH MAX_DEPTH UP TO 25 ##

# Hyperopt best results:

#         Mean AMS: 3.733631
#         Var AMS: 0.141557
#         Mean threshold: 0.871343
#         Var threshold: 0.002815
#         Trial: 60

# RESULTS, WITH BEST CV THRESHOLD

# Train AMS: 8.474507
# Test AMS: 3.477927
# Leaderboard AMS: 3.401259

# * Real difference between mean CV AMS of 3.73 and test AMS of 3.47: too much
#   randomization and variance in extra trees?
# * Very high score in train (8.47) way above 5: big overfitting caused by too much
#   depth in the trees?


## WITH MAX_DEPTH UP TO 15 ##

# Hyperopt best results:
#         Mean AMS: 3.478727
#         Var AMS: 0.013807
#         Mean threshold: 0.753908
#         Var threshold: 0.003377
#         Trial: 116

# RESULTS, WITH BEST CV THRESHOLD

# Scores with best threshold in CV:
# Train AMS: 3.647394
# Test AMS: 3.345966
# Leaderboard AMS: 3.314367

# * Same problem as 1. with depth 25, difference between CV and test AMS
# * No noticeable overfitting though, but lower test scores compared to depth 25.
