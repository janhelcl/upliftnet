"""
Keras Model with PCG loss function and metric
"""
from typing import Tuple

import tensorflow as tf

import metrics


class UplitNet(tf.keras.Model):
    """
    """        
    def train_step(self, batch: Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]):
        """        
        Performs one step in optimizing the uplift neural net model weights 
        based on the LambdaLoss paradigm and Promotional Cumulative Gain metric.
        
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
    
    def compile(self, optimizer=None, loss=None, metrics=None, **kwargs):
        """
        Compiles the model with PCG as default loss and metric
        """
        if loss is None:
            loss = ApproxPCGLoss()
        if metrics is None:
            metrics = [AveragePromotionalCumulativeGain()]
        super().compile(optimizer=optimizer, loss=loss, metrics=metrics, **kwargs)
    
    
class AveragePromotionalCumulativeGain(tf.keras.metrics.Metric):
    """
    Average Promotional Cumulative Gain
    
    Since target transformation is happening on the level of individual batches there is
    no straightforward way of combining the results across batches and computing one meaningful
    PCG for the whole epoch. Average of PCGs of inidividual batches is computed instead.
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
        if sample_weight is not None:
            sample_weight = tf.squeeze(sample_weight)
        self.gains.assign_add(metrics.pcg_tf_metric(y, uplift, sample_weight))
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

@tf.function
def _flip_and_scale(y_true: tf.Tensor, treatment: tf.Tensor) -> tf.Tensor:
    """
    Prepares target for uplift ranking
    
    :param y_true: label, typically response idicator or revenue
    :param treatment: binary treatment indicator, 1 stands for treatment, 0 for control
    
    :returns: transformed target
    """
    return tf.where(tf.cast(treatment, tf.bool),
                    tf.divide(y_true, tf.reduce_sum(treatment)),
                    tf.negative(tf.divide(y_true, tf.reduce_sum(tf.negative(treatment-1))))
                   )
