"""
Utilities for ploting uplift related charts.
"""
from typing import Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import auc
from sklearn.utils.validation import check_consistent_length

from upliftnet import metrics
from upliftnet import utils


def cumulative_gain_plot(y_true: np.ndarray,
                         uplift: np.ndarray,
                         treatment: np.ndarray,
                         weights: np.ndarray = None,
                         model_name: str = 'Uplift Model',
                         figsize: Tuple[int, int] = (10,6)
                        ) -> plt.Figure:
    """
    Plots the cumulative gain curve
    
    :param indices: x axis of the plot
    :param curve_values: y axis of the plot
    
    :returns: cgc plot
    """
    indices, curve_values = metrics.cumulative_gain_curve(
        y_true, uplift, treatment, weights
    )
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.plot(indices, curve_values, label=f'{model_name}, auc: {round(auc(indices, curve_values), 4)}')
    ax.plot([0, 1], [0, curve_values[-1]], linestyle="--", label="Random Model", color="black")
    ax.set_xlabel("% of treated subjects")
    ax.set_ylabel("Cumulative Gain")
    ax.set_title("Cumulative Gain Curve")
    ax.legend()
    return ax


def target_plot(y_true: np.ndarray,
                uplift: np.ndarray,
                treatment: np.ndarray,
                bins: int = 10,
                figsize: Tuple[int, int] = (10,6)
               ) -> plt.Figure:
    """Plots mean response in control/treatment group by uplift score quantile bins
    
    :param y_true: label, typically response idicator or revenue
    :param uplift: model predictions used for ranking, higher value means higher expected uplift
    :param treatment: binary treatment indicator, 1 stands for treatment, 0 for control
    :param weights: sample weights
    
    :returns: target plot
    """
    y_true, uplift, treatment= np.array(y_true), np.array(uplift), np.array(treatment)
    check_consistent_length(y_true, uplift, treatment)
    utils.check_is_binary(treatment)
    
    results = pd.DataFrame({
        'y_true': y_true,
        'uplift': uplift,
        'treatment': treatment
    })
    results['quantile'] = pd.qcut(-results['uplift'], bins, labels=False, duplicates='drop')
    results['treatment'] = results['treatment'].map({
        1: 'treatment',
        0: 'control'
    })
    plt.figure(figsize=figsize)
    fig = sns.barplot(x='quantile',
                      y='y_true',
                      hue='treatment',
                      errorbar=('ci', False),
                      estimator=np.mean,
                      data=results
                     )
    fig.axhline(np.mean(results[results['treatment']=='treatment']['y_true']),
                linestyle='--',
                color='k',
                alpha=0.5,
                label='treatment avg')
    fig.axhline(np.mean(results[results['treatment']=='control']['y_true']),
                linestyle='--',
                color='r',
                alpha=0.5,
                label='control avg')
    plt.title('Mean response by quantile bin')
    plt.legend()
    return fig
    

def true_lift_plot(y_true: np.ndarray,
                   uplift: np.ndarray,
                   treatment: np.ndarray,
                   weights: np.ndarray = None,
                   bins: int = 10,
                   figsize: Tuple[int, int] = (10,6)
                  ) -> plt.Figure:
    """Plots true lift by uplift score quantile bins
    
    Plots true lift across quantile bins as defined in "The True Lift Model" by Victor S.Y. Lo available at:
    https://www.researchgate.net/publication/220520042_The_True_Lift_Model_-_A_Novel_Data_Mining_Approach_to_Response_Modeling_in_Database_Marketing
    
    :param y_true: label, typically response idicator or revenue
    :param uplift: model predictions used for ranking, higher value means higher expected uplift
    :param treatment: binary treatment indicator, 1 stands for treatment, 0 for control
    :param weights: sample weights
    
    :returns: true lift plot
    """
    if weights is None:
        weights = np.ones_like(y_true)
    y_true, uplift, treatment, weights = np.array(y_true), np.array(uplift),\
                                        np.array(treatment), np.array(weights)
    check_consistent_length(y_true, uplift, treatment, weights)
    utils.check_is_binary(treatment)
    
    results = pd.DataFrame({
        'y_true_weight': y_true * weights,
        'uplift': uplift,
        'treatment': treatment,
        'weights': weights
    })
    results['quantile'] = pd.qcut(-results['uplift'], bins, labels=False, duplicates='drop')
    grouped_treatment = results[results['treatment']==1].groupby('quantile').sum()
    grouped_control = results[results['treatment']==0].groupby('quantile').sum()
    increment_model = grouped_treatment['y_true_weight'] / grouped_treatment['weights'] - \
                      grouped_control['y_true_weight'] / grouped_control['weights']
    treatment_sum = results[results['treatment']==1].sum(axis=0)
    control_sum = results[results['treatment']==0].sum(axis=0)
    increment_random = treatment_sum['y_true_weight'] / treatment_sum['weights'] - \
                       control_sum['y_true_weight'] / control_sum['weights']
    true_lift = increment_model - increment_random
    return true_lift.plot.bar(figsize=figsize, title='True lift by uplift score quantile bins')


def calibration_plot(y_true: np.ndarray,
                     uplift: np.ndarray,
                     treatment: np.ndarray,
                     weights: np.ndarray = None,
                     bins: int = 10,
                     bin_type: str = 'quantile',
                     figsize: Tuple[int, int] = (10,10)
               ) -> plt.Figure:
    """Plots the observed uplift against predicted uplift
    
    :param y_true: label, typically response idicator or revenue
    :param uplift: model predictions used for ranking, higher value means higher expected uplift
    :param treatment: binary treatment indicator, 1 stands for treatment, 0 for control
    :param weights: sample weights
    
    :returns: calibration plot
    """
    if weights is None:
        weights = np.ones_like(y_true)
    y_true, uplift, treatment, weights = np.array(y_true), np.array(uplift),\
                                        np.array(treatment), np.array(weights)
    check_consistent_length(y_true, uplift, treatment, weights)
    utils.check_is_binary(treatment)
    valid_bin_types = ['quantile', 'uniform']
    assert bin_type in valid_bin_types, f'Invalid bin_type, must be one of {valid_bin_types}'
    
    results = pd.DataFrame({
        'y_true_weight': y_true * weights,
        'uplift_weight': uplift * weights,
        'uplift': uplift,
        'treatment': treatment,
        'weights': weights
    })
    if bin_type == 'quantile':
        results['bin'] = pd.qcut(results['uplift'], bins, labels=False, duplicates='drop')
    else:
        results['bin'] = pd.cut(results['uplift'], bins, labels=False, duplicates='drop')
    grouped_treatment = results[results['treatment']==1].groupby('bin').sum()
    grouped_control = results[results['treatment']==0].groupby('bin').sum()
    mean_true_uplift = grouped_treatment['y_true_weight'] / grouped_treatment['weights'] - \
                       grouped_control['y_true_weight'] / grouped_control['weights']
    grouped = results.groupby('bin')[['uplift_weight', 'weights']].sum()
    mean_predicted_uplift = grouped['uplift_weight'] / grouped['weights']
    mean_true_uplift.name = 'actual'
    mean_predicted_uplift.name = 'predicted'
    res = mean_true_uplift.to_frame().join(mean_predicted_uplift)
    res = res.sort_index()
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(res['predicted'], res['actual'],
            label='Model',
            marker='s'
           )
    plt.xlabel('Predicted uplift')
    plt.ylabel('Actual uplift')
    min_ax = min(mean_true_uplift.min(), mean_predicted_uplift.min())
    max_ax = max(mean_true_uplift.max(), mean_predicted_uplift.max())
    plt.plot([min_ax, max_ax+1], [min_ax, max_ax+1],
             color='k', label='Perfectly calibrated',
             linestyle="--"
            )
    plt.legend()
    return fig
