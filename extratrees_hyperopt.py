"""Hyperparameter search with hyperopt on Higgs ML Data for extra trees"""

import os
import argparse
import time
import pickle

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer

from hyperopt import Trials, fmin, hp, tpe, STATUS_OK, space_eval
#@bugfix: Fixes a bug from Hyperopt.fmin with BSON
from hyperopt import base
base.have_bson = False

from utils import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-evals', default=100, type=int, help='number of trials for hyperopt search')
    parser.add_argument('--seed', type=int, help='random generators seed (default: None)')
    parser.add_argument('--logdir', type=str, default='./trials', help='save directory')
    args = parser.parse_args()
    return args

# define hyperparameters space
PARAMS_SPACE = {
    'n_estimators': hp.choice('n_estimators', range(10, 201)),
    'criterion': hp.choice('criterion', ["gini", "entropy"]),
    'max_depth': hp.choice('max_depth', range(3, 15)),
    'min_samples_split': hp.choice('min_samples_split', range(2, 21)),
    'min_samples_leaf': hp.choice('min_samples_leaf', range(1, 11)),
    'max_features': hp.uniform('max_features', 0.15, 1),
    'bootstrap': hp.choice('bootstrap', [True, False]),
    'class_weight': hp.choice('class_weight', [None, 'balanced', 'balanced_subsample']),
    'n_jobs': -1
}

if __name__ == '__main__':

    args = parse_args()
    if args.seed is not None:
        # random seed for reproducibility
        np.random.seed(args.seed)

    df_train, df_lead, df_test, _, = get_cern_datasets()

    imputer = SimpleImputer(strategy='median')
    # prepare datasets
    data, target, weights = prepare_cern_dataset(df_train, True, imputer)
    # test_data, test_target, test_weights = prepare_cern_dataset(df_test, False, imputer)
    # lead_data, lead_target, lead_weights = prepare_cern_dataset(df_lead, False, imputer)

    k_folds = StratifiedKFold(n_splits=5, shuffle=True) # CV scheme

    def hyperopt_func(params):
        """Function to optimize with hyperopt
        Args:
            params (dict): dictionary defining the parameter space to search in
        Returns:
            (float): the mean AMS score over all folds (maximized AMS)
        """
        cv_ams = list() # max ams scores
        cv_thresholds = list() # maximizing thresholds
        cv_ams_curves = list() # ams curves
        # iterate over folds

        for train_idx, val_idx in k_folds.split(data, target):

            # need to scale the weights to keep normalized
            train_ratio = len(train_idx)/len(data)
            train_data, val_data = data.loc[train_idx, :], data.loc[val_idx, :]
            train_target, val_target = target[train_idx], target[val_idx]
            train_weights, val_weights = weights[train_idx]/train_ratio, weights[val_idx]/(1 - train_ratio)

            # train a model on this fold
            clf = ExtraTreesClassifier(**params)
            clf.fit(train_data, train_target)
            # predict on the validation set
            preds = clf.predict_proba(val_data)[:, 1]

            # compute AMS, threshold
            thresholds = np.linspace(0, 1, 500)
            ams_scores, _ = ams_curve(val_target, preds, val_weights, thresholds)
            cv_ams_curves.append(ams_scores)
            ams_max, th_max = max_ams((ams_scores, thresholds))
            cv_ams.append(ams_max)
            cv_thresholds.append(th_max)
# WITH MAX
        # compute mean metrics over folds
        ams, ams_var = np.mean(cv_ams), np.var(cv_ams)
        threshold, threshold_var = np.mean(cv_thresholds), np.var(cv_thresholds)
        # return objective (in 'loss' key), status, plus all the useful information
        return {'loss': -ams, 'status': STATUS_OK,
                'loss_variance': ams_var,
                'threshold': threshold, 'threshold_variance': threshold_var,
                'cv_ams_curves': cv_ams_curves,
                }

    print("------------------------------------")
    print("Beginning of hyperopt process")
    start = time.time()
    trials = Trials()
    bparams = fmin(hyperopt_func, PARAMS_SPACE, algo=tpe.suggest,
                   max_evals=args.max_evals, trials=trials)

    # map indices to values for parameters with search space defined with hp.choice()
    bparams = space_eval(PARAMS_SPACE, bparams)
    print("------------------------------------")
    print("Done")
    print("The best hyperparameters are: ", "\n")
    print(bparams)
    end = time.time()
    print('Time elapsed: {}s'.format(end - start))
    # save Trials object in log dir
    if args.logdir is not None:
        with open(os.path.join(args.logdir, 'Trials-extratrees.pkl'), 'wb') as file:
            pickle.dump(trials, file, pickle.HIGHEST_PROTOCOL)


## WITH MAX_DEPTH UP TO 15 ##

# The best hyperparameters are:

# {'bootstrap': True, 'class_weight': None, 'criterion': 'entropy', 'max_depth': 13,
#  'max_features': 0.7969006970974817, 'min_samples_leaf': 2, 'min_samples_split': 7,
#  'n_estimators': 191, 'n_jobs': -1}
# Time elapsed: 1:31:46


## WITH MAX_DEPTH UP TO 25 ##

# The best hyperparameters are:

# {'bootstrap': False, 'class_weight': 'balanced', 'criterion': 'entropy',
#   'max_depth': 23, 'max_features': 0.6716967767567084, 'min_samples_leaf': 8,
#   'min_samples_split': 12, 'n_estimators': 184, 'n_jobs': -1}
# Time elapsed: 2:01:19
