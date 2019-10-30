"""Submission on Higgs ML Data with XGBOOST"""

import os
import argparse
import pickle

import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import pandas as pd
import xgboost as xgb

from hyperopt import space_eval

from xgboost_hyperopt import PARAMS_SPACE
from utils import *


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

    df_train, df_test = get_kaggle_datasets()

    # prepare datasets
    train_data, train_target, train_weights = prepare_kaggle_dataset(df_train, True)
    test_data = prepare_kaggle_dataset(df_test, False)

    # xgboost DMatrices
    dtrain = xgb.DMatrix(train_data, label=train_target)
    dtest = xgb.DMatrix(test_data)

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

    bparams = trials.argmin
    bparams = space_eval(PARAMS_SPACE, bparams)
    # retrain the model with best set of parameters on full training set
    num_round = int(bparams.pop('num_boost_round'))
    print('Training on train set...')
    gbm = xgb.train(bparams, dtrain, num_round)
    print('Done\n')

    # predictions with best threshold on the test set
    test_preds = gbm.predict(dtest)

    # Write submission to a file
    rank_orders = np.argsort(test_preds) + 1
    print('Test preds scores sorted: ', test_preds[rank_orders-1])
    print('Test predictions sorted:', round_predictions(test_preds[rank_orders-1], threshold))
    df_submission = pd.DataFrame({'EventId': df_test['EventId'],
                                  'RankOrder': rank_orders,
                                  'Class': ['s' if y == 1 else 'b'
                                            for y in round_predictions(test_preds, threshold)]})
    df_submission.to_csv('submissions/xgboost_submission.csv', index=False)


    # solution_path = 'data/solution_from_cern.csv'
    # submission_path = 'submissions/xgboost_submission.csv'

    # from HiggsBosonCompetition_AMSMetric_rev1 import AMS_metric
    # AMS_metric(solution_path, submission_path)

    # From the AMS_metric() in HiggsBosonCompetition_AMSMetric_rev1.py:

    # signal = 387.2166747044417, background = 6193.463527669711
    # AMS = 4.866424406557141
    # => The scores are wrong, they are way too high!!

    # The submission on Kaggle showed only the score in private leaderboard, which is
    # the same as ours. Therefore we haven't made any errors in our evaluations.

    # RANK AGAINST THE PARTICIPANTS ON KAGGLE (exluding the late submissions we can't see):

    # On private leaderboard: 786 / 1785
    # On public leaderboard: 858 / 1785
