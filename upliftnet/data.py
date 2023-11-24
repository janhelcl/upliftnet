"""
Utility for generating synthetic data for testing
"""
from typing import List, Tuple

import numpy as np


def generate_logistic_data(treatment_coefs: List[float],
                           control_coefs: List[float],
                           n_treatment: int,
                           n_control: int,
                           mu: np.ndarray = None,
                           cov: np.ndarray = None
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """Creates dataset based on logistic models
    
    Generates dataset of multinomial normal distributed features
    with logistic response based on two different logit models for
    treatment and control groups. Returns X - a matrix of features, 
    treatment indicator and y - observed values generated from Bernouli
    distribution.
    
    :param treatment_coefs: coefficients of treatment model (intercept first)
    :param control_coefs: coefficients of control model (intercept first)
    :param n_treatment: size of treatment group
    :param n_control: size of control group
    :param mu: mean of the multinomial distribution (vector of 0s by default)
    :param cov: covariance matrix of the distribution (identity by default)
    
    :returns: X, y, treatment
    """
    
    if mu is None:
        mu = np.zeros(len(treatment_coefs)-1)
    if cov is None:
        cov = np.eye(len(treatment_coefs)-1)
        
    assert len(treatment_coefs) == len(control_coefs), \
            'Treatment and control coefs must be of same lenght'
    assert len(mu) == len(treatment_coefs)-1, \
            'mu must be of same lenght as coefs'
    assert cov.shape == (len(treatment_coefs)-1, len(treatment_coefs)-1), \
            'Covariance matrix must be of shape (len(mu), len(mu))'
                        
    X = np.random.multivariate_normal(mu, cov, n_treatment + n_control)
    # add column vector of 1s for intercept
    X_ = np.hstack(
        (np.ones(n_treatment + n_control).reshape(-1, 1), X)
    )
    # generate the logistic response
    y_treatment = 1 / (1 + np.exp(-X_[:n_treatment].dot(np.array(treatment_coefs))))
    y_control = 1 / (1 + np.exp(-X_[n_treatment:].dot(np.array(control_coefs))))
    # generate the treatment indicator
    treatment = np.hstack(
        (np.ones(n_treatment), np.zeros(n_control))
    )
    y_latent = np.hstack((y_treatment, y_control))
    y = np.random.binomial(n=1, p=y_latent)
    return X.astype(np.float32), y.astype(np.float32), treatment.astype(np.float32)
