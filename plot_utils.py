import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
sns.set()

from utils import *


def plot_roc_curve(roc_curve, show_auc=False, ax=None):
    """Plots ROC curve showing AUC"""
    tpr, fpr, thresholds = roc_curve
    if ax in None:
        ax = plt.gca()
    legend = None
    if show_auc:
        auc = metrics.auc(fpr, tpr)
        legend = 'AUC = {.3f}'.format(auc)
    ax.plot(fpr, tpr, label=legend)

    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    if legend is not None:
        ax.legend(loc='lower right')
    return ax


def plot_ams_curve(ams_curve, show_max=False, ax=None):
    """Plots curves of AMS scores for threshold"""
    ams_scores, thresholds = ams_curve
    if ax is None:
        ax = plt.gca()
    legend = None
    if show_max: # show maximum in legend
        ams, threshold = max_ams(ams_curve)
        legend = 'Max = {:.3f}'.format(ams)
    ax.plot(thresholds, ams_curve)

    ax.set_xlabel('Decision threshold')
    ax.set_ylabel('AMS score')
    if lenged is not None:
        ax.legend(loc='best')


def plot_decision_matrix(cm, classes=['s', 'b'], normalize=False, ax=None):
    """Plot the decision matrix"""
    if normalize:
        cm = cm / np.sum(cm)[:, None]
    if ax is None:
        ax = plt.gca()
    fmt = '.2f' if normalize else 'd'
    cm = pd.DataFrame(cm, index=classes, columns=classes)
    heatmap = sns.heatmap(cm, annot=True, fmt=fmt)

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return ax


def plot_ams_with_trials(trials, title=None, ax=None):
    """Plot AMS found at each trial and the best one"""
    all_ams = [-ams for ams in trials.losses()]
    if ax is None:
        ax = plt.gca()
    if title is None:
        title = 'AMS Score with Trials'
    ax.set_title(title)
    ax.scatter(np.arange(1, len(trials.trials)+1), all_ams, marker='x')
    bams = max(all_ams)
    ax.axhline(bams, linestyle='--', c='C2', label='Max = {:.3f}'.format(bams))
    ax.set_xlabel('Trial t')
    ax.set_ylabel('Score')
    ax.legend()
    return ax

def plot_cv_ams_curves_for_trial(ams_curves, title=None, ax=None):
    """Plot curves of AMS for threshold for each CV fold"""
    if ax is None:
        ax = plt.gca()
    if title is None:
        title = 'CV AMS Curves'

    ax.set_title(title)
    for k in range(5):
        ax.plot(np.linspace(0, 1, 500), ams_curves[k], label='fold{}'.format(k+1))

    ax.set_xlabel('Decision threshold')
    ax.set_ylabel('Score')
    ax.legend(loc='best')
    return ax

def plot_ams_curves_on_train_test(ams_curve_train, ams_curve_test, title=None, ax=None):
    """Plot curves of AMS for threshold on train and test set"""
    if ax is None:
        ax = plt.gca()
    if title is None:
        title = 'AMS Curves On Train And Test Set'

    ax.set_title(title)
    ax.plot(np.linspace(0, 1, 500), ams_curve_train, label='train')
    ax.plot(np.linspace(0, 1, 500), ams_curve_test, label='test')
    ax.set_ylabel('Decision threshold')
    ax.set_xlabel('Score')
    ax.legend(loc='best')
    return ax
