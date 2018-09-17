from datetime import datetime
import imp
import os

import pandas as pd
from pytz import timezone
from sklearn.externals import joblib


def save_search_result(search_obj, dataset, estimator_type, results_dir='./results/'):
    """Saves a GridSearchCV or RandomizedSearchCV result to a file"""
    results = search_obj.cv_results_
    score = search_obj.best_score_
    tz = timezone('US/Eastern')
    date = datetime.now(tz).isoformat(timespec='minutes', sep='_')
    filename = '%.3f_%s_%s_%s' % (score, dataset, estimator_type, date)
    joblib.dump(results, results_dir + filename + '.pkl')


def load_best_result(dataset, estimator_type, results_dir='./results/'):
    """Loads the best (highest scoring) result for a previously-executed grid/randomized search"""
    results_files = [fn for fn in os.listdir(results_dir) if estimator_type in fn and dataset in fn]
    best_results_file = sorted(results_files)[-1]
    return joblib.load(results_dir + best_results_file)


def refresh_import(modname):
    """Reload a module and return the fresh module"""
    file, pathname, description = imp.find_module(modname)
    mod = imp.load_module(modname, file, pathname, description)
    imp.reload(mod)
    return mod


def scikit_cv_result_to_df(cv_res):
    """Convert a GridSearchCV.cv_result_ dictionary to a dataframe indexed by hyperparameter

    :param cv_res: cv_result_ object (attribute of a GridSearchCV/RandomizedSearchCV instance)
    :type cv_res: dict
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
    return cv_res_df
