"""
# TODO: docs
"""
from typing import Tuple

import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import auc
from sklearn.utils.extmath import stable_cumsum
from sklearn.utils.validation import check_consistent_length


def cumulative_gain_curve(y_true: np.ndarray,
                          uplift: np.ndarray,
                          treatment: np.ndarray,
                          weights: np.ndarray = None
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the cumulative gain curve
    
    Shows the cumulative difference in target variable between treatment and control groups.
    Inspired by this awesome jupyter notebook based book full of memes by Matheus Facure
    available at: https://matheusfacure.github.io/python-causality-handbook/19-Evaluating-Causal-Models.html
    (simplified to binary treatment case).
    
    :param y_true: label, typically response idicator or revenue
    :param uplift: model predictions used for ranking, higher value means higher expected uplift
    :param treatment: binary treatment indicator, 1 stands for treatment, 0 for control
    :param weights: sample weights
    
    :returns: indices, cumulative curve values
    """
    if weights is None:
        weights = np.ones_like(y_true)
    y_true, uplift, treatment, weights = np.array(y_true), np.array(uplift),\
                                        np.array(treatment), np.array(weights)
    check_consistent_length(y_true, uplift, treatment, weights)
    _check_is_binary(treatment)
    
    desc_score_indices = np.argsort(uplift)[::-1]
    y_true, uplift, treatment, weights = y_true[desc_score_indices], uplift[desc_score_indices],\
                                        treatment[desc_score_indices], weights[desc_score_indices]
    
    y_true_ctrl, y_true_trmnt = y_true.copy(), y_true.copy()
    
    y_true_trmnt[treatment == 0] = 0
    y_true_ctrl[treatment == 1] = 0
    
    size = len(y_true)
    num_all = stable_cumsum(weights)
    
    num_trmnt = stable_cumsum(treatment * weights)
    y_trmnt = stable_cumsum(y_true_trmnt * weights)
    
    num_ctrl = num_all - num_trmnt
    y_ctrl = stable_cumsum(y_true_ctrl * weights)
    
    indices = num_all / np.sum(weights)
    curve_values = (np.divide(y_trmnt, num_trmnt, out=np.zeros_like(y_trmnt), where=num_trmnt != 0) -
                    np.divide(y_ctrl, num_ctrl, out=np.zeros_like(y_ctrl), where=num_ctrl != 0)
                   ) * indices
    #prepend zeros to ensure the curve starts at [0, 0]
    return np.r_[0., indices], np.r_[0., curve_values]


def cgc_auc(y_true: np.ndarray,
            uplift: np.ndarray,
            treatment: np.ndarray,
            weights: np.ndarray = None
           ) -> float:
    """
    Computes Area Under Cumulative Gain Curve
    
    :param y_true: label, typically response idicator or revenue
    :param uplift: model predictions used for ranking, higher value means higher expected uplift
    :param treatment: binary treatment indicator, 1 stands for treatment, 0 for control
    :param weights: sample weights
    
    :returns: cgc auc
    """
    return auc(*cumulative_gain_curve(y_true, uplift, treatment, weights))


def pcg_xgb_metric(preds: np.ndarray, 
                   dtrain: 'xgboost.DMatrix'
                  ) -> Tuple[str, float]:
    """
    Promotional Cumulative Gain metric for xgb early stopping
    
    NOTE: xgb early stopping assumes minimization of a loss,
    here we want to maximize this quantity therefore the
    negative of pcg is returned
    
    :param preds: predictions of xgb model
    :param dtrain: DMatrix with properly transformed label
    
    :returns: 'pcg', -pcg
    """
    labels = dtrain.get_label()
    ranks = scipy.stats.rankdata(-preds, method='ordinal')
    n = len(labels)
    return 'pcg', -(labels*(n-ranks+1)).sum()


@tf.function
def pcg_tf_metric(y: tf.Tensor,
                  uplift: tf.Tensor,
                  sample_weight: tf.Tensor = None,
                  normalize: bool = True
                 ) -> tf.Tensor:
    """
    TensorFlow implementation of PCG metric
    
    :param y: properly transformed target
    :param uplift: model output
    :param weights: sample weights
    :param normalize: whether to normalize the results
    
    :returns: PCG
    """
    if sample_weight is None:
        sample_weight = tf.ones_like(y)
    argsorted = tf.argsort(uplift, direction='DESCENDING')
    inds = tf.argsort(argsorted)
    cumsum_w = tf.cumsum(tf.gather(sample_weight, argsorted))
    ranks = tf.gather(cumsum_w, inds) - sample_weight/2. + .5
    promotions = tf.reduce_sum(sample_weight) - tf.cast(ranks, dtype=tf.float32) + 1.
    promotions *= sample_weight
    if normalize:
        promotions /= tf.reduce_sum(promotions)
    return tf.reduce_sum(y * promotions)

