"""Run hyperparameter tuning with all predictors, optimized for use on HPC."""

import pickle
import logging
import uuid
from argparse import ArgumentParser, RawTextHelpFormatter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.metrics import cohen_kappa_score, make_scorer

import src.helpers as hlp
from src.analysis import mlhelpers as mlh

# Define path to outputs
project_dir = Path(__file__).parent.parent.resolve()
results_path = project_dir / 'results' / 'hyperparam_search'

# Get datetime for file naming
datetime_str = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
hex_str = uuid.uuid4().hex[0:6]

# Load cross-validation indices
with open(
    project_dir / 'data' / 'misc' / 'vtdata_10f_spatial_cv_indices.pkl',
    'rb'
) as file:
    split_idx = pickle.load(file)

# Define common model scores
scores = {
    'cohen_kappa': make_scorer(cohen_kappa_score),
    'accuracy': 'accuracy',
    'f1_macro': 'f1_macro'
}


def get_parser():
    """Get parser object for this script."""

    parser = ArgumentParser(
        description=__doc__,
        formatter_class=RawTextHelpFormatter
    )

    parser.add_argument(
        "-clf", "--classifier",
        help="Classifier to use, 'rf' or 'mlp'",
        action="store",
        choices=('rf', 'mlp'),
        dest="classifier",
        required=False,
        type=str,
        default='rf'
    )

    parser.add_argument(
        "-n", "--n_iterations",
        help="Number of random hyperparameter permutations",
        action="store",
        dest="n_iterations",
        required=False,
        type=int,
        default=10
    )

    parser.add_argument(
        "-e", "--epochs",
        help="Number of epochs for the 'mlp' classifier",
        action="store",
        dest="epochs",
        required=False,
        type=int,
        default=500
    )

    parser.add_argument(
        "-l", "--use-log-file",
        help="Redirect output stream to logfile?",
        action='store_true',
        dest="log",
        default=False,
        required=False
    )

    return parser


def output_to_logfile(clf: str) -> logging.Logger:
    """Define logging options"""

    log_dir_path = project_dir / 'logs'

    if not log_dir_path.is_dir():
        log_dir_path.mkdir()

    log_file_path = log_dir_path / \
        f'{clf}_hyperp_full_{datetime_str}_{hex_str}.log'

    log_level = logging.DEBUG
    logger = hlp.setup_logging(log_file_path, log_level)

    return logger


def tune_random_forest(n_permutations: int) -> None:
    """Tune hyperparameters of a Random Forest classifier"""

    from sklearn.ensemble import RandomForestClassifier

    data_path = project_dir / 'data' / 'interim' / 'vtdata_full.pkl'
    data_df = pd.read_pickle(data_path)

    vt_X = data_df.drop(columns=["x", "y", "plot_id", "vt"])
    vt_y = data_df["vt"]

    n_features = vt_X.shape[1]

    # Tuneable parameters and their ranges
    params_rf = {
        'n_estimators': stats.randint(50, 500),
        'criterion': ('gini', 'entropy'),
        'class_weight': ('balanced', None),
        'max_depth': stats.randint(12, 30),
        'max_features': stats.randint(
            int(np.sqrt(n_features))-3,
            int(np.sqrt(n_features))+3
        )
    }

    # Initialize classifier
    random_forest = RandomForestClassifier(oob_score=True, n_jobs=-1)

    rf_rand_hypertune_cv = mlh.hyperparam_search_cv(
        classifier=random_forest,
        param_vals=params_rf,
        target_scores=scores,
        cv_indices=split_idx,
        n_jobs=1,
        verbose=3,
        method='random',
        n_random_iters=n_permutations,
        refit=False,
        return_train_score=True
    )

    # Fit classifier
    history = rf_rand_hypertune_cv.fit(vt_X, vt_y)

    # Save cross-validation results as csv
    rf_results_df = pd.DataFrame(rf_rand_hypertune_cv.cv_results_)

    result_csv_file_path = results_path / \
        f'RF_10fCV_n{n_permutations}_rands_full_{datetime_str}_{hex_str}.csv'

    rf_results_df.to_csv(
        result_csv_file_path,
        index=False
    )

    print(
        f"Finished Random Forest tuning! Results in: {result_csv_file_path}"
    )

    return history


def tune_perceptron(n_permutations: int, epochs: int = 500):
    """Tune multi-layer-perceptron classifier"""

    import tensorflow as tf

    # Load data
    data_path = project_dir / 'data' / 'processed'
    feat_path_mlp = data_path / 'vt_X_scaled_and_dummies.pkl'
    vt_path_fact = data_path / 'vt_y_fact.pkl'

    vt_X = pd.read_pickle(feat_path_mlp).drop(columns=["x", "y", "plot_id"])
    vt_y = pd.read_pickle(vt_path_fact)

    n_features = vt_X.shape[1]
    n_targets = len(set(vt_y.values.flatten()))

    # Tuneable parameters and their ranges
    params_mlp = {
        'n_hidden_layers': (1, 2),
        'kernel_regularize': (True, False),
        'alpha_l1': stats.uniform(loc=0.0001, scale=0.0099),
        'dropout_rate': (None, 0.1, 0.2, 0.3),
        'learning_rate': stats.uniform(loc=0.001, scale=0.009),
        'n_nodes_l1': (16, 32, 64, 128),
        'n_nodes_l2': (16, 32, 64, 128),
        'optim_name': ["sgd", "nadam", "rmsprop"]
    }

    # Initialize classifier, compatible with sklearn
    mlp_skl = tf.keras.wrappers.scikit_learn.KerasClassifier(
        build_fn=mlh.build_mlp_classifier,
        epochs=epochs,
        verbose=0,
        n_features=n_features,
        n_targets=n_targets
    )

    mlp_rand_hypertune_cv = mlh.hyperparam_search_cv(
        classifier=mlp_skl,
        param_vals=params_mlp,
        target_scores=scores,
        cv_indices=split_idx,
        n_jobs=-1,
        verbose=1,
        method='random',
        n_random_iters=n_permutations,
        refit=False,
        return_train_score=True
    )

    # Instantiate callback to reduce training time if appropriate
    callback = tf.keras.callbacks.EarlyStopping(
        monitor='accuracy',  # If 'monitor' metric...
        min_delta=0.0001,  # ...does not improve by 'min_delta'...
        patience=50  # ...after 'patience' epochs, stop training.
    )

    history = mlp_rand_hypertune_cv.fit(
        vt_X,
        vt_y,
        callbacks=[callback]
    )

    # Save cv-results as csv
    mlp_results_df = pd.DataFrame(mlp_rand_hypertune_cv.cv_results_)

    result_csv_file_path = results_path / \
        f'MLP_10fCV_n{n_permutations}_rands_full_{datetime_str}_{hex_str}.csv'

    mlp_results_df.to_csv(
        result_csv_file_path,
        index=False
    )

    print(
        f"Finished MLP tuning! Results in: {result_csv_file_path}"
    )

    return history


def main():
    """Run analysis"""

    # Parse command line arguments
    args = get_parser().parse_args()

    # Instantiate machine class
    classifier = args.classifier
    n_iterations = args.n_iterations
    epochs = args.epochs

    # Log results?
    if args.log:
        output_to_logfile(clf=classifier)

    if classifier == 'rf':
        _ = tune_random_forest(n_permutations=n_iterations)

    if classifier == 'mlp':
        _ = tune_perceptron(n_permutations=n_iterations, epochs=epochs)


if __name__ == "__main__":

    main()
