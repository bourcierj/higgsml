"""Inference on Higgs ML Data with Adaboost"""

import os
import argparse
import pickle

import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
from hyperopt import space_eval
from sklearn.impute import SimpleImputer

from adaboost_hyperopt import PARAMS_SPACE
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

    imputer = SimpleImputer(strategy='median')
    # prepare datasets
    train_data, train_target, train_weights = prepare_cern_dataset(df_train, True, imputer)
    test_data, test_target, test_weights = prepare_cern_dataset(df_test, False, imputer)
    lead_data, lead_target, lead_weights = prepare_cern_dataset(df_lead, False, imputer)

    # load Trials object from hyperopt search in log dir
    with open(os.path.join(args.logdir, 'Trials-adaboost.pkl'), 'rb') as file:
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

    # plot ams with trials
    fig, ax = plt.subplots(figsize=(9,6))
    plot_ams_with_trials(trials, ax=ax)
    plt.savefig('figures/adaboost/AMS Score with Trials.svg')
    plt.show()

    # plot ams curves for each fold
    fig, ax = plt.subplots(figsize=(9,6))
    plot_cv_ams_curves_for_trial(bresult['cv_ams_curves'], ax=ax)
    plt.savefig('figures/adaboost/CV AMS Curves.svg')
    plt.show()

    bparams = trials.argmin
    bparams = space_eval(PARAMS_SPACE, bparams)
    # retrain the model with best set of parameters on full training set
    print('Training on train set...')
    clf = AdaBoostClassifier(**bparams)
    clf.fit(train_data, train_target)
    print('Done\n')

    # predictions with best threshold
    print('Scores with best threshold in CV:')
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



# Hyperopt best results:

#     Mean AMS: 2.976480
#     Var AMS: 0.013250
#     Mean threshold: 0.514228
#     Var threshold: 0.000003
#     Trial: 10

# RESULTS, WITH BEST CV THRESHOLDS

