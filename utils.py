import numpy as np
import pandas as pd


def get_cern_datasets():
    """Reads CERN csv file and splits in into four datasets: train, Kaggle public
    eaderboard, Kaggle private leaderboard, and unused."""
    df = pd.read_csv('data/atlas-higgs-challenge-2014-v2.csv', sep=',')
    df = df.astype({'Label': 'category', 'KaggleSet': 'category'})
    df_train = df[df["KaggleSet"] == 't']
    df_public_lead = df[df["KaggleSet"] == 'b']
    df_private_lead = df[df["KaggleSet"] == 'v']
    df_unused = df[df["KaggleSet"] == 'u']
    return df_train, df_public_lead, df_private_lead, df_unused


def get_kaggle_datasets():
    """Reads Kaggle csv files to return training dataset (250k events) and test dataset
    (550k events)
    """
    df_train = pd.read_csv('data/training.csv', sep=',')
    df_train = df_train.astype({'Label': 'category'})
    df_test = pd.read_csv('data/test.csv', sep=',')
    assert(len(df_train) == 250000)
    assert(len(df_test) == 550000)
    return df_train, df_test


def prepare_cern_dataset(df, is_train=True, imputer=None):
    """Prepares dataset into data, target and weight"""

    # get features data, weights and target variable
    data = df.drop(['EventId', 'Label', 'Weight', 'KaggleWeight', 'KaggleSet'], axis=1)
    kaggle_weights = df['KaggleWeight']
    # encode categorical class variable
    target = df['Label'].cat.codes
    # replace -990.0 values with NaNs
    data = data.replace(-999.0, np.NaN)
    # impute missing values
    if imputer:
        if is_train:
            imputer.fit(data.values)

        data = pd.DataFrame(imputer.transform(data.values), columns=data.columns)

    return data, target, kaggle_weights


def prepare_kaggle_dataset(df, is_train=True, imputer=None):
    """Prepares Kaggle dataset, use for creating submission"""
    if is_train:
        data = df.drop(['EventId', 'Label', 'Weight'], axis=1)
        kaggle_weights = df['Weight']
        # encode categorical class variable
        target = df['Label'].cat.codes
        # replace -990.0 values with NaNs
        data = data.replace(-999.0, np.NaN)
        # impute missing values
        if imputer:
            imputer.fit(data)
            data = pd.DataFrame(imputer.transform(data), columns=data.columns)
        return data, target, kaggle_weights
    else:
        data = df.drop(['EventId'], axis=1)
        # replace -990.0 values with NaNs
        data = data.replace(-999.0, np.NaN)
        # impute missing values
        if imputer:
            data = pd.DataFrame(imputer.transform(data), columns=data.columns)
        return data


def ams_score(target, pred, weights):
    """Approximate Median Significance defined as:
        AMS = sqrt(
                2 { (s + b + b_r) log[1 + (s/(b+b_r))] - s}
              )
    where b_r = 10, b = background, s = signal, log is natural logarithm.
    s and b are the unnormalised true positive and false positive rates,
    respectively, weighted by the weights of the dataset.
    """
    # true positive rate, weighted
    s = weights.dot(np.logical_and(pred == 1, target == 1))
    # false positive rate, weighted
    b = weights.dot(np.logical_and(pred == 1, target == 0))

    br = 10.0
    radicand = 2 *((s+b+br) * np.log(1.0 + s/(b+br)) - s)
    if radicand < 0:
        raise Exception("Radicand is negative.")
    else:
        return np.sqrt(radicand)


def tp_fp_score(target, pred):
    """Un-normalized true positive rate and false positive rate"""
    tp = np.logical_and(pred == 1, target == 1)
    fp = np.logical_and(pred == 1, target == 0)

    return tp, fp


def round_predictions(pred_scores, threshold):
    """Rounds predictions for threshold t such that for each pred p if p > t
    round to 1 else 0"""
    pred = np.where(pred_scores > threshold, 1, 0)
    return pred


def ams_curve(target, pred_scores, weights, thresholds):
    """Computes AMS scores on dataset as a function of the decision threshold"""
    ams_scores = [ams_score(target, round_predictions(pred_scores, t), weights) for t in thresholds]
    return ams_scores, thresholds


def max_ams(curve):
    """Max of AMS score and threshold argmax"""
    ams_scores, thresholds = curve
    argmax = np.argmax(ams_scores)
    return (ams_scores[argmax], thresholds[argmax])
