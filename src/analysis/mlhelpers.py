"""
Helper functions to handle machine learning classification.
"""

from typing import Any, Literal, Optional
import tensorflow as tf
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def hyperparam_search_cv(
    classifier: Any, param_vals: dict,
    target_scores: dict, cv_indices: list,
    n_jobs: int = -1, verbose: int = 3,
    method: Literal['grid', 'random'] = "random",
    n_random_iters: int = 50, refit: bool = False,
    return_train_score: bool = True
) -> Any:
    '''
    Perform randomized search or grid search to optimize model
    hyperparameters.

    Arguments:
    'method': 'grid' -> GridSearchCV(), 'random' -> RandomizedSearchCV()
    See scikit-learn function documentation for input parameter details.

    'param_vals': dictionary defining hyperparameters to tune and their
    potential ranges.

    'target_scores': dict specifying classification metrics to use.

    'cv_indices': list of indices to use in cross-validation.

    'n_random_iters': Number of iterations if 'method'=='random'

    'refit': Refit classifier to best parameters?

    'return_train_score': include scores for training data in result.

    Return:
    'split_idx' --> Split indices.
    '''

    valid_methods = ('grid', 'random')

    if method not in valid_methods:
        raise ValueError(f"'method' must be one of {valid_methods}.")

    # GRID SEARCH
    if method == "grid":
        return GridSearchCV(
            classifier, param_grid=param_vals,
            scoring=target_scores, cv=cv_indices,
            n_jobs=n_jobs, verbose=verbose, refit=refit,
            return_train_score=return_train_score
        )

    # RANDOM SEARCH
    if method == "random":
        return RandomizedSearchCV(
            classifier, param_distributions=param_vals,
            scoring=target_scores, cv=cv_indices,
            n_iter=n_random_iters, refit=refit,
            verbose=verbose, return_train_score=return_train_score
        )


def build_mlp_classifier(
    n_features: int, n_targets: int,
    kernel_regularize: bool = False, alpha_l1: float = 0.0001,
    alpha_l2: float = 0.0001, dropout_rate: Optional[float] = None,
    learning_rate: float = 0.001, n_hidden_layers: int = 1,
    n_nodes_l1: int = 64, n_nodes_l2: int = 64, n_nodes_l3: int = 64,
    optim_name: Literal['nadam', 'sgd', 'rmsprop'] = "nadam"
) -> tf.keras.Sequential:
    """
    Build a feed-forward neural network with chosen hyperparameters.
    """

    valid_optimizers = ('nadam', 'sgd', 'rmsprop')

    if optim_name not in valid_optimizers:
        raise ValueError(
            f"Input error! 'optim_name' must be one of {valid_optimizers}"
        )

    if n_hidden_layers < 1 or n_hidden_layers > 3:
        raise ValueError(
            "Input error! 'n_hidden_layers' must be 1, 2, or 3!"
        )

    # Keras sequential, feed-forward neural network
    mlp_clf = tf.keras.Sequential()

    if kernel_regularize:
        kernel_reg = tf.keras.regularizers.l2(
            l2=alpha_l1
        )

    # Input layer + first hidden
    mlp_clf.add(tf.keras.layers.Dense(
        n_nodes_l1,
        input_dim=n_features,
        kernel_regularizer=kernel_reg if kernel_regularize else None,
        activation='tanh'
    ))

    # Add dropout
    if dropout_rate:
        mlp_clf.add(tf.keras.layers.Dropout(dropout_rate))

    # Add additional hidden layers
    for idx in range(n_hidden_layers-1):

        # Add kernel regularization
        if kernel_regularize:
            kernel_reg = tf.keras.regularizers.l2(
                l2=alpha_l1 if (idx == 0) else alpha_l2
            )

        mlp_clf.add(tf.keras.layers.Dense(
            n_nodes_l2 if (idx == 0) else n_nodes_l3,
            kernel_regularizer=kernel_reg if kernel_regularize else None,
            activation='tanh'
        ))

        if dropout_rate:
            mlp_clf.add(tf.keras.layers.Dropout(dropout_rate))

    # Output layer
    mlp_clf.add(tf.keras.layers.Dense(n_targets, activation='softmax'))

    # Compile model and define optimizer, loss, and metrics
    if optim_name == "sgd":
        optim = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    if optim_name == "nadam":
        optim = tf.keras.optimizers.Nadam(learning_rate=learning_rate)
    if optim_name == "rmsprop":
        optim = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

    mlp_clf.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        optimizer=optim,
        metrics=['accuracy']
    )

    return mlp_clf
