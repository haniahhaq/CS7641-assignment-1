from datetime import datetime
import os
from pathlib import Path

import numpy as np
import pandas as pd
from pytz import timezone
from sklearn.externals import joblib
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.utils import compute_sample_weight


def save_search_result(search_obj, dataset, estimator_type, results_dir='./results/'):
    """Saves a GridSearchCV or RandomizedSearchCV result to a file"""
    results = search_obj.cv_results_
    score = search_obj.best_score_
    tz = timezone('US/Eastern')
    date = datetime.now(tz).isoformat(timespec='minutes', sep='_').replace(':', '-')
    filename = 'search_%s_%s_%.3f_%s' % (dataset, estimator_type, score, date)
    path = Path(results_dir + filename + '.pkl')
    joblib.dump(results, path)


def save_learning_curve(dataset, learner_type, train_sizes, train_mean, train_std, test_mean, test_std, results_dir='./results/'):
    learning_result = {
        'train_sizes': train_sizes,
        'train_mean': train_mean,
        'train_std': train_std,
        'test_mean': test_mean,
        'test_std': test_std,
    }
    score = np.max(learning_result['test_mean'])
    tz = timezone('US/Eastern')
    date = datetime.now(tz).isoformat(timespec='minutes', sep='_').replace(':', '-')
    filename = 'learning_%s_%s_%.3f_%s' % (dataset, learner_type, score, date)
    path = Path(results_dir + filename + '.pkl')
    joblib.dump(learning_result, path)


def load_best_search(dataset, estimator_type, results_dir='./results/'):
    """Loads the best (highest scoring) result for a previously-executed grid/randomized search"""
    results_files = [fn for fn in os.listdir(results_dir)
                     if fn.startswith('search') and
                     estimator_type in fn and dataset in fn]
    best_results_file = sorted(results_files)[-1]
    print('Found the following results files for this dataset/algorithm: %s' % results_files)
    print('Returning results for the highest-scoring-file: %s' % best_results_file)
    return joblib.load(results_dir + best_results_file)


def load_best_learning(dataset, estimator_type, results_dir='./results/'):
    """Loads the best (highest scoring) result for a previously-executed learning curve"""
    results_files = [fn for fn in os.listdir(results_dir)
                     if fn.startswith('learning') and
                     estimator_type in fn and dataset in fn]
    best_results_file = sorted(results_files)[-1]
    print('Found the following results files for this dataset/algorithm: %s' % results_files)
    print('Returning results for the highest-scoring-file: %s' % best_results_file)
    return joblib.load(results_dir + best_results_file)


def scikit_cv_result_to_df(cv_res, drop_splits=True):
    """Convert a GridSearchCV.cv_result_ dictionary to a dataframe indexed by hyperparameter

    :param cv_res: cv_result_ object (attribute of a GridSearchCV/RandomizedSearchCV instance)
    :type cv_res: dict
    :param drop_splits: whether to drop columns with individual cross-validation scores (default: True)
    :type drop_splits: bool
    :return: DataFrame with a MultiIndex corresponding to each parameter
    :rtype: pd.DataFrame
    """
    params = [k for k in cv_res.keys() if k.startswith('param_')]
    params_to_shorthand = { # Mapping type to remove the 'param_' prefix
        p : p[6:] for p in params
    }
    cv_res_df = pd.DataFrame(cv_res, columns=[k for k in cv_res.keys() if k != 'params'])
    cv_res_df = cv_res_df.rename(index=str, columns=params_to_shorthand)
    cv_res_df = cv_res_df.set_index(list(params_to_shorthand.values()))
    if drop_splits:
        cv_res_df = cv_res_df.drop(axis=1, labels=[c for c in cv_res_df.columns if c.startswith('split')])
    return cv_res_df


# This code copied from https://github.com/JonathanTay/CS-7641-assignment-1/blob/master/helpers.py
def balanced_accuracy(truth,pred):
    wts = compute_sample_weight('balanced',truth)
    return accuracy_score(truth,pred,sample_weight=wts)

balanced_accuracy_scorer = make_scorer(balanced_accuracy)
