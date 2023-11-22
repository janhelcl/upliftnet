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


class UplitRankNet(tf.keras.Model):
    """
    Custom model implementing LambdaRANK algorithm with PCG as the optimized metric
    
    Implements the ideas from Learning to Rank for Uplift Modeling paper using a neural net
    instead of gradient boosting (see: https://arxiv.org/pdf/2002.05897.pdf). The main motivation
    for switching to NNs was computational. Unlike in classical learning to rank problems all
    data points here are considered part of one single query. At every iteration of the LambdaRANK
    algorithm a metric is computed for each pair of obseravations in each query. This becomes 
    infeasible when we have only one query with hundreds of thousands or even millions of observations.
    NNs naturally solve this problem by using mini-batches during training, so the quadratic number
    of metrics get computed only within very small subsets of data. This would be cumbersome
    to implement in gradient boosting.
    
    The true power of LambdaRANK lies in its ability to optimize a non-smooth cost function via standard
    gradient descent. This way we can (almost) directly optimize the Area under Cumulative Gain Curve,
    our ultimate metric of interest. For efficiency reasons (to avoid repeated sorting and computation
    of cumulative sums) a combination of target transformation and a new metric, Promotional Cumulative
    Gain, are introduced. These two tricks together mimic the behavior of Area under CGC while being
    computationaly much cheaper.
    
    The target transformation is taken care of inside this class, the sign of 'y_true' is flipped for the
    control group and both treatment and control groups are rescaled by their respective sizes. The goal
    of the algorithm is to rank the observations by this resulting quantity. Imagine two observations very
    similar (or even identical) in X, one belonging to treatment group the other in control. Since 'treatment'
    is assumed to be independent of X, the first observation will push both of these up the list while the
    second will push both of them down, so in some sense they both end up somewhere in the final list based on
    their difference in 'y_true'. This, simpified to its core, is how this model optimizes uplift.
    
    This class can be used the same way as its superclass 'tf.keras.Model' with both "Functional API" or
    by subclassing (see: https://www.tensorflow.org/api_docs/python/tf/keras/Model). There are few limitations
    however:
        - the output layer is assumed to be 'tf.keras.layers.Dense(1, activation='linear')'
        - the data is required to be in a specific format, param 'x' must be 'tf.data.Dataset((X, y_true, treatment))'
    """        
    def train_step(self, batch: Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]):
        """
        Computes and applies LambdaRANK gradients with PCG as the optimized metric
        
        Performs one step in optimizing the uplift neural net model weights 
        based on the LambdaRANK paradigm and Promotional Cumulative Gain metric.
        
        :param batch: tuple that gets yielded by tf.data.Dataset, must contain three
        or four tensors in this order: X, y_true, treatment and (optionally) sample weight
        
        :returns: dictionary of metrics results
        """
        if len(batch) == 4:
            X, y_true, treatment, sample_weight = batch
        else:
            X, y_true, treatment = batch
            sample_weight = None
        # transform the target to the required form: flip sign for cg and scale
        # note that the normalization needs to be done within each batch
        y = _flip_and_scale(y_true, treatment)
        return super().train_step((X, y, sample_weight))
    
    def test_step(self, data: Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]):
        """
        Evaluates the model on given dataset
        
        :param data: tuple that gets yielded by tf.data.Dataset, must contain three
        or four tensors in this order: X, y_true, treatment and (optionally) sample weight
        
        :returns: dictionary of metrics results
        """
        if len(data) == 4:
            X, y_true, treatment, sample_weight = data
        else:
            X, y_true, treatment = data
            sample_weight = None
        y = _flip_and_scale(y_true, treatment)
        return super().test_step((X, y, sample_weight))

    def build_graph(self, input_shape):
        """
        Utility for ploting model architecture
        
        :param input_shape: input shape
        """
        x = tf.keras.layers.Input(shape=input_shape, name='input')
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
    
    
class AveragePromotionalCumulativeGain(tf.keras.metrics.Metric):
    """
    Average Promotional Cumulative Gain
    
    Since target transformation is happening on the level of individual batches there is
    no straightforward way of combining the results across batches and computing one meaningful
    PCG for the whole epoch. Average of PCGs of inidividual batches is computed instead.
    
    Note that the value of PCG is not standardized in any way and so is not comparable
    accros datasets of different sizes even if comming from the same distribution.
    """
    def __init__(self, name='avg_promotional_cumulative_gain', **kwargs):
        super(AveragePromotionalCumulativeGain, self).__init__(name=name, **kwargs)
        self.gains = self.add_weight(name='pcg', initializer='zeros')
        self.counter = self.add_weight(name='counter', initializer='zeros')
        
    def update_state(self, y, uplift, sample_weight=None):
        """
        Computes PCG for the given batch
        
        :param y: properly transformed target
        :param uplift: model output
        :param sample_weight: sample weights
        """
        y = tf.squeeze(y)
        uplift = tf.squeeze(uplift)
        sample_weight = tf.squeeze(sample_weight)
        self.gains.assign_add(pcg_tf_metric(y, uplift, sample_weight))
        self.counter.assign_add(1)

    def result(self):
        return self.gains / self.counter
    
    
@tf.function
def approx_ranks(logits, weights=None):
    """Weighted version of tfr.losses_impl.approx_ranks().
    """
    list_size = tf.shape(logits)[1]
    x = tf.tile(tf.expand_dims(logits, 2), [1, 1, list_size])
    y = tf.tile(tf.expand_dims(logits, 1), [1, list_size, 1])
    pairs = tf.sigmoid(y - x)
    if weights is not None:
        wy = tf.tile(tf.expand_dims(weights, 1), [1, list_size, 1])
        pairs = pairs * wy
    return tf.reduce_sum(input_tensor=pairs, axis=-1) + .5


class ApproxPCGLoss(tf.keras.losses.Loss):
    """
    Approximate Promotional Cumulative Gain Loss
    
    ApproxPCGLoss implemented in the LambdaLoss framework introduced by google,
    see https://github.com/tensorflow/ranking or https://research.google/pubs/pub47258/
    """    
    def __init__(self, name='ApproxPCGLoss', normalize=True, temperature=1., **kwargs):
        super(ApproxPCGLoss, self).__init__(name=name, **kwargs)
        self.normalize = normalize
        self.temperature = temperature
        
    def __call__(self, y, uplift, sample_weight=None):
        uplift /= self.temperature
        if sample_weight is None:
            sample_weight = tf.ones_like(y)
        y = tf.squeeze(y)
        uplift = tf.squeeze(uplift)
        sample_weight = tf.squeeze(sample_weight)
        y, uplift, sample_weight = tf.expand_dims(y, axis=0), tf.expand_dims(uplift, axis=0), tf.expand_dims(sample_weight, axis=0)
        ranks = approx_ranks(uplift, sample_weight)
        promotions = tf.reduce_sum(sample_weight) - tf.cast(ranks, dtype=tf.float32) + 1.
        promotions *= sample_weight
        if self.normalize:
            promotions /= tf.reduce_sum(promotions)
        pcg = tf.reduce_sum(y * promotions)
        return -pcg

# @tf.function
# def _flip_and_scale(y_true: tf.Tensor, treatment: tf.Tensor) -> tf.Tensor:
#     """
#     Prepares target for uplift ranking
    
#     :param y_true: label, typically response idicator or revenue
#     :param treatment: binary treatment indicator, 1 stands for treatment, 0 for control
    
#     :returns: transformed target
#     """
#     return tf.where(tf.cast(treatment, tf.bool),
#                     tf.divide(y_true, tf.reduce_sum(treatment)),
#                     tf.negative(tf.divide(y_true, tf.reduce_sum(tf.negative(treatment-1))))
#                    )

@tf.function
def _flip_and_scale(y_true: tf.Tensor, treatment: tf.Tensor) -> tf.Tensor:
    """
    Prepares target for uplift ranking
    
    :param y_true: label, typically response idicator or revenue
    :param treatment: binary treatment indicator, 1 stands for treatment, 0 for control
    
    :returns: transformed target
    """
    return tf.where(tf.cast(treatment, tf.bool),
                    y_true,
                    -y_true
                   )
