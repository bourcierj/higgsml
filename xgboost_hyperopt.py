"""Hyperparameter search with hyperopt on Higgs ML Data for XGBOOST"""

import os
import argparse
import time
import pickle

import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold

from hyperopt import Trials, fmin, hp, tpe, STATUS_OK, space_eval
#@bugfix: Fixes a bug from Hyperopt.fmin with BSON
from hyperopt import base
base.have_bson = False

from utils import *

# command: ipython xgboost_higgs.py -- --max-evals 100 --seed 42 --logdir './trials'
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-evals', default=100, type=int, help='number of trials for hyperopt search')
    parser.add_argument('--seed', type=int, help='random generators seed (default: None)')
    parser.add_argument('--logdir', type=str, default='./trials', help='save directory')
    args = parser.parse_args()
    return args

# define hyperparameters space
PARAMS_SPACE = {
    'objective': 'binary:logistic',
    'metric': 'auc',
    'num_boost_round': hp.choice('num_boost_round', range(10, 101)),
    'eta': hp.uniform('eta', 0.1, 0.6),
    'max_depth': hp.choice('max_depth',  range(3, 11)),
    'min_child_weight': hp.quniform('min_child_weight', 0.7, 1, 0.05),
    'subsample': hp.quniform('subsample', 0.7, 1, 0.05),
    'gamma': hp.quniform('gamma', 0, 1, 0.05),
    'colsample_bytree': hp.quniform('colsample_bytree', 0.7, 1, 0.05),
    'lambda': hp.quniform('lambda', 1, 2, 0.05),
}

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
    data, target, weights = prepare_cern_dataset(df_train)
    # test_data, test_target, test_weights = prepare_cern_dataset(df_test)
    # lead_data, lead_target, lead_weights = prepare_cern_dataset(df_lead)

    # xgboost DMatrices
    dtrain = xgb.DMatrix(data, label=target)

    #@problemfix: We did not shuffle the splits when creating the stratifie-k-fold
    # object and between trials: hence the folds when trying parameters where always the
    # same, and a consequence we over-parameters on this cross-val setup.
    # Now we make sure to have different, stochastic folds between trials.

    k_folds = StratifiedKFold(n_splits=5, shuffle=True) # CV scheme

    def hyperopt_func(params):
        """Function to optimize with hyperopt
        Args:
            params (dict): dictionary defining the parameter space to search in
        Returns:
            (float): the mean AMS score over all folds (maximized AMS)
        """
        cv_ams = list()  # max ams scores
        cv_thresholds = list()  # maximizing thresholds
        cv_ams_curves = list()  # ams curves
        cv_xgb_ntrees = list()  # optimal number of trees (early stopping on validation set)
        # iterate over folds
        num_round = int(params.pop('num_boost_round'))

        for train_idx, val_idx in k_folds.split(data, target):

            # need to scale the weights to keep normalized
            train_ratio = len(train_idx)/len(data)
            train_data, val_data = data.loc[train_idx, :], data.loc[val_idx, :]
            train_target, val_target = target[train_idx], target[val_idx]
            train_weights, val_weights = weights[train_idx]/train_ratio, weights[val_idx]/(1 - train_ratio)
            dtrain = xgb.DMatrix(train_data, label=train_target)
            dval = xgb.DMatrix(val_data, label=val_target)

            dwatch = [(dval, 'val')] # watch validation set during training for early stopping
            # train an XGBOOST model on this fold, with early stopping
            gbm = xgb.train(params, dtrain, num_round, early_stopping_rounds=20, evals=dwatch, verbose_eval=False)
            # predict on the validation set with optimal number of trees
            ntree_limit = 0 if not hasattr(gbm, 'best_iteration') else gbm.best_iteration
            preds = gbm.predict(dval, ntree_limit=ntree_limit)
            cv_xgb_ntrees.append(ntree_limit if ntree_limit > 0 else num_round)
            # compute AMS, threshold
            thresholds = np.linspace(0, 1, 500)
            ams_scores, _ = ams_curve(val_target, preds, val_weights, thresholds)
            cv_ams_curves.append(ams_scores)
            ams_max, th_max = max_ams((ams_scores, thresholds))
            cv_ams.append(ams_max)
            cv_thresholds.append(th_max)

        # compute mean metrics over folds
        ams, ams_var = np.mean(cv_ams), np.var(cv_ams)
        threshold, threshold_var = np.mean(cv_thresholds), np.var(cv_thresholds)
        xgb_ntree, xgb_ntree_var = np.mean(cv_xgb_ntrees), np.var(cv_xgb_ntrees)
        # return objective (in 'loss' key), status, plus all the useful information
        return {'loss': -ams, 'status': STATUS_OK,
                'loss_variance': ams_var,
                'threshold': threshold, 'threshold_variance': threshold_var,
                'cv_ams_curves': cv_ams_curves,
                'xgb_ntree': xgb_ntree,
                'xgb_ntree_variance': xgb_ntree_var
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
        with open(os.path.join(args.logdir, 'Trials-xgb.pkl'), 'wb') as file:
            pickle.dump(trials, file, pickle.HIGHEST_PROTOCOL)


## NEW RESULTS ##

# Beginning of hyperopt process
# 100%|████████| 100/100 [1:20:58<00:00, 48.59s/it, best loss: -3.601115172402216]
# ------------------------------------
# Done
# The best hyperparameters are:

# {'colsample_bytree': 1.0, 'eta': 0.19719016533324585, 'gamma': 0.25, 'lambda': 1.5,
#   'max_depth': 7, 'metric': 'auc', 'min_child_weight': 0.75, 'num_boost_round': 76,
#   'objective': 'binary:logistic', 'seed': 42, 'subsample': 1.0}

# Time elapsed: 4858.59530544281s

## OLD RESULTS ##

# Beginning of hyperopt process
# 100%|█████████████████████████████████████████████| 100/100 [51:40<00:00, 31.00s/it, best loss: -3.615105370547847]
# ------------------------------------
# Done
# The best hyperparameters are:

# {'colsample_bytree': 0.8500000000000001, 'eta': 0.11270558036451875,
#  'gamma': 0.8500000000000001, 'lambda': 1.05, 'max_depth': 5, 'min_child_weight': 0.9,
#  'num_boost_round': 48, 'subsample': 0.8500000000000001}
# Time elapsed: 3100.5009088516235s
