"""Helper functions for processing the VT distribution ML workflow"""

import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import GroupKFold


def get_cv_indices(X, y, method="kfold-cv", n_splits=10, groups=None):
    '''
    Split the data according to desired subsampling technique.

    Arguments:
    X --> feature matrix
    y --> targets
    'method' --> Split method. Values: 'kfold-cv', 'LGO-kfold-cv'
    'n_split' --> Number of splits (integer).
    'groups' --> Variable to group by, e.g. PlotID.

    Return:
    'split_idx' --> Split indices.
    '''

    # Create KFold split indices for data
    if method == "kfold-cv":
        kf = KFold(n_splits=n_splits, shuffle=True)
        split_idx = kf.split(X)

    if method == "LGO-kfold-cv":
        group_kfold = GroupKFold(n_splits=n_splits)
        split_idx = group_kfold.split(X, y, groups)

    return split_idx


def generate_vt_spatialcv_idxgen(n_splits=10, group_by="plot_id"):
    '''
    Helper function to create indexes with standard version of data.
    '''

    data_df = pd.read_pickle("../data/interim/vtdata.pkl")
    data_df = data_df.sample(frac=1, random_state=7).reset_index(drop=True)

    vt_X = data_df.drop(columns="vt")
    vt_y = data_df["vt"]

    groups = vt_X[group_by]

    # Get kfold indices -- Leave group out cross validation
    split_idx = get_cv_indices(vt_X, vt_y,
                               method="LGO-kfold-cv", n_splits=n_splits,
                               groups=groups)
    return split_idx


def hyperparam_search_cv(
    classifier, param_vals, target_scores, cv_indices,
    n_jobs=6, verbose=3, method="grid", n_random_iters=50,
    refit=True, return_train_score=True
):
    '''
    Perform randomized search or grid search for optimizing ML model
    hyperparameters.

    Arguments:
    'method' --> grid->GridSearchCV, random->RandomizedSearchCV
    See scikit-learn function documentation for input parameter details.

    Return:
    'split_idx' --> Split indices.
    '''

    if method != "grid" and method != "random":
        raise ValueError("'method' must be 'grid' or 'random'.")

    # Import sklearn functions
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

    # GRID SEARCH
    if method == "grid":
        return GridSearchCV(
            classifier, param_grid=param_vals,
            scoring=target_scores, cv=cv_indices,
            n_jobs=n_jobs, verbose=verbose, refit=refit,
            return_train_score=return_train_score
        )

    # RANDOM SEARCH
    elif method == "random":
        return RandomizedSearchCV(
            classifier, param_distributions=param_vals,
            scoring=target_scores, cv=cv_indices,
            n_iter=n_random_iters, refit=refit,
            verbose=verbose, return_train_score=return_train_score
        )


def save_obj_pkl(obj, name, path='../results/'):
    '''
    Save python objects in pickle format.
    '''
    import pickle

    if name.split(".")[-1] != "pkl" or len(name.split(".")) == 1:
        name = name + ".pkl"

    with open(path + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)


def load_obj_pkl(name, path='../results/'):
    '''
    Load python objects in pickle format.
    '''

    import pickle

    with open(path + name + '.pkl', 'rb') as f:
        return pickle.load(f)
