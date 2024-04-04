'''
Helper functions to handle/determine variable correlation.
'''

import scipy.stats as stat
import pandas as pd
import numpy as np


def cramers_corrected_stat(confusion_matrix: pd.DataFrame) -> np.float64:
    """
    Calculate Cramers V statistic for categorial-categorial
    variable association. Uses correction from Bergsma and Wicher,
    Journal of the Korean Statistical Society 42 (2013): 323-328
    """

    # Calculate parameters
    chi2 = stat.chi2_contingency(confusion_matrix)[0]
    n_obs = confusion_matrix.sum().sum()
    phi2 = chi2/n_obs
    n_rows, n_columns = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((n_columns-1)*(n_rows-1))/(n_obs-1))
    rcorr = n_rows - ((n_rows-1)**2)/(n_obs-1)
    kcorr = n_columns - ((n_columns-1)**2)/(n_obs-1)

    return np.round(np.sqrt(phi2corr / min((kcorr-1), (rcorr-1))), decimals=3)
