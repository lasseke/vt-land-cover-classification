"""Tune NN with BayesCV, optimized for use on HPC."""

import dill as pickle
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.model_selection import cross_validate

import tensorflow as tf
import tensorflow_addons as tfa

import src.helpers as hlp

# Define path to outputs
project_dir = Path(__file__).parent.parent.resolve()
results_path = project_dir / 'results' / 'hyperparam_search'

# Get datetime for file naming
datetime_str = datetime.now().strftime("%y%m%d_%H%M%S")


def output_to_logfile() -> logging.Logger:
    """Define logging options"""

    log_dir_path = project_dir / 'logs'

    if not log_dir_path.is_dir():
        log_dir_path.mkdir()

    log_file_path = log_dir_path / \
        f'nn_hyperp_bayes_{datetime_str}.log'

    log_level = logging.DEBUG
    logger = hlp.setup_logging(log_file_path, log_level)

    return logger


def main() -> None:

    global build_ANN_model  # Needs to be global for pickling for some reason

    output_to_logfile()

    # Load CV indices
    with open(
        project_dir / 'data' / 'misc' / 'vtdata_5f_spatial_cv_indices.pkl',
        'rb',
    ) as file:
        split_indices = pickle.load(file)

    # Read data
    vt_y_nn = pd.read_pickle(
        project_dir / 'data' / 'processed' / "vt_y_fact.pkl"
    )
    vt_X_nn = pd.read_pickle(
        project_dir / 'data' / 'processed' / \
            "vt_X_scaled_and_dummies_allbands.pkl"
    ).drop(columns=["x","y","plot_id"])

    # Get number of features and targets
    n_X_cols = vt_X_nn.shape[1]
    n_y_classes = len(set(vt_y_nn["vt_integer"]))

    def build_ANN_model(
            alpha_l1=0.001, alpha_l2=0.001,
            dropout_rate=0.1, #learning_rate=0.01,
            n_nodes_l1=64, n_nodes_l2=64, # n_nodes_l3=64,
            optim_name="nadam", n_hidden_layers=1,
    ) -> tf.keras.Sequential:
        '''Defining Sequential() architecture'''

        # Define the keras sequential NN
        nn = tf.keras.Sequential()
        
        # Input + hidden layer
        nn.add(
            tf.keras.layers.Dense(
                n_nodes_l1,
                input_dim=n_X_cols,
                kernel_regularizer=tf.keras.regularizers.L1L2(
                    l1=alpha_l1,
                    l2=alpha_l2
                ),
                activation='tanh',
        ))
        nn.add(tf.keras.layers.Dropout(dropout_rate))
        
        # Hidden additional layers
        for _ in range(n_hidden_layers-1):
            nn.add(
                tf.keras.layers.Dense(
                    n_nodes_l2, 
                    kernel_regularizer=tf.keras.regularizers.L1L2(
                        l1=alpha_l1, l2=alpha_l2
                    ),
                    activation='tanh'
                )
            )

            nn.add(tf.keras.layers.Dropout(dropout_rate))

        # nn.add(tf.keras.layers.Dropout(dropout_rate))

        # Output layer
        nn.add(tf.keras.layers.Dense(n_y_classes, activation='softmax'))
        
        # Compile model and define optimizer, loss, and metrics
        if optim_name == "sgd":
            optim = tf.keras.optimizers.SGD()
        elif optim_name == "nadam":
            optim = tf.keras.optimizers.Nadam()
        elif optim_name == "rmsprop":
            optim = tf.keras.optimizers.RMSprop()
        else:
            optim = tf.keras.optimizers.RMSprop()

        nn.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=optim,
            metrics=[
                tf.keras.metrics.SparseCategoricalAccuracy(
                    name="accuracy",
                ),
                tfa.metrics.CohenKappa(
                    num_classes=n_y_classes,
                    name='cohen_kappa',
                    sparse_labels=True,
                ),
            ],
        )
        
        return nn
    
    # CV search
    early_stopper = tf.keras.callbacks.EarlyStopping(
        monitor="accuracy",
        min_delta=0.001,
        patience=10,
        verbose=1,
        mode="auto",
        baseline=None,
        restore_best_weights=True,
        start_from_epoch=0,
    )

    nn_sklearn = tf.keras.wrappers.scikit_learn.KerasClassifier(
        build_fn=build_ANN_model,
        epochs=100,
        callbacks=[early_stopper],
        workers=20,
        use_multiprocessing=True,
        verbose=1,
    )

    optimized_nn = BayesSearchCV(
        estimator=nn_sklearn,
        search_spaces={
            'n_hidden_layers': Integer(low=1, high=2, prior='uniform'),
            'dropout_rate': Real(low=0.01, high=0.2, prior='uniform'),
            'n_nodes_l1': Integer(low=16, high=64, prior='uniform'),
            'n_nodes_l2': Integer(low=16, high=64, prior='uniform'),
            'optim_name': Categorical(['sgd', 'nadam', 'rmsprop']),
            'alpha_l1': Real(low=0.0001, high=0.01, prior='log-uniform'),
            'alpha_l2': Real(low=0.0001, high=0.01, prior='log-uniform'),
        },
        n_jobs=10,
        n_iter=50,
        cv=split_indices,
        random_state=221,
        verbose=2,
        return_train_score=True,
    )

    # Fit to data
    optimized_nn.fit(
        vt_X_nn.astype('float32'),
        vt_y_nn.astype('int'),
    )

    # Save cv results as backup
    pd.DataFrame(optimized_nn.cv_results_).to_csv(
        project_dir / 'results' / 'hyperparam_search' / "nn_5fcv_bayes_n50_optim.csv"
    )

    try:
        with open(
            project_dir / 'results' / 'hyperparam_search' / \
                "nn_5fcv_bayes_n50_optim.pkl",
            'wb',
        ) as pkl_file:
            pickle.dump(
                optimized_nn,
                pkl_file,
            )
    except Exception as e:
        print(e)

if __name__ == "__main__":

    main()
