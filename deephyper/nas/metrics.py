"""This module provides different metric functions. A metric can be defined by a keyword (str) or a callable. If it is a keyword it has to be available in ``tensorflow.keras`` or in ``deephyper.netrics``. The loss functions availble in ``deephyper.metrics`` are:
* Sparse Perplexity: ``sparse_perplexity``
* R2: ``r2``
* AUC ROC: ``auroc``
* AUC Precision-Recall: ``aucpr``
"""
import functools
from collections import OrderedDict


import tensorflow as tf
from deephyper.core.utils import load_attr

from sklearn.metrics import f1_score


def r2(y_true, y_pred):
    SS_res = tf.math.reduce_sum(tf.math.square(y_true - y_pred), axis=0)
    SS_tot = tf.math.reduce_sum(
        tf.math.square(y_true - tf.math.reduce_mean(y_true, axis=0)), axis=0
    )
    output_scores = 1 - SS_res / (SS_tot + tf.keras.backend.epsilon())
    r2 = tf.math.reduce_mean(output_scores)
    return r2


def mae(y_true, y_pred):
    return tf.keras.metrics.mean_absolute_error(y_true, y_pred)


def mse(y_true, y_pred):
    return tf.keras.metrics.mean_squared_error(y_true, y_pred)


def rmse(y_true, y_pred):
    return tf.math.sqrt(tf.math.reduce_mean(tf.math.square(y_pred - y_true)))


def acc(y_true, y_pred):
    return tf.keras.metrics.categorical_accuracy(y_true, y_pred)

def f1(y_true, y_pred):
    y_pred_binary = tf.round(y_pred)
    y_true = tf.cast(y_true, tf.float32)
    
    # True positives
    true_positive = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 1), tf.equal(y_pred_binary, 1)), tf.float32))
    # False positives
    false_positive = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 0), tf.equal(y_pred_binary, 1)), tf.float32))
    # False negatives
    false_negative = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 1), tf.equal(y_pred_binary, 0)), tf.float32))
    
    # Precision and Recall calculations
    precision = true_positive / (true_positive + false_positive + tf.keras.backend.epsilon())
    recall = true_positive / (true_positive + false_negative + tf.keras.backend.epsilon())
    
    # F1 score calculation
    f1score = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
    return f1score
   
    # print("Start f1")
    # print(type(y_true))
    # print(y_true.shape)
    # for i in range(len(y_true)):
    #     true = y_true[i]
    #     pred = y_pred[i]
    #     if true == pred == 1:
    #         tp += 1
    #     elif pred == 1 and true != pred:
    #         fp += 1
    #     elif pred == 0 and true != pred:
    #         fn += 1
    
    # print('fp, fn')
    # print(type(fn))
    # print(type(y_pred))

    # precision = true_positives / (tp + fp) if (tp + fp) > 0 else 0
    # recall = true_positives / (tp + fn) if (tp + fn) > 0 else 0
    # print('presicion recall')

    # return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    # # # return f1_score(y_true, y_pred)
    # # # metr = tf.keras.metrics.F1Score(threshold=0.5)
    # # # metr.update_state(y_true, y_pred)
    # # # return metr.result()


    # y_pred = tf.round(y_pred)  # Assuming y_pred comes from a sigmoid activation for binary classification
    
    # # Calculate Precision and Recall
    # precision = tf.metrics.Precision()
    # recall = tf.metrics.Recall()

    # # Update states of the metrics
    # precision.update_state(y_true, y_pred)
    # recall.update_state(y_true, y_pred)
    
    # # Use TensorFlow operations to calculate F1 score
    # p = precision.result()
    # r = recall.result()
    
    # return 2 * (p * r) / (tf.add(p, r) + tf.keras.backend.epsilon())  # Use tf.keras.backend.epsilon() to avoid division by zero

    # y_pred = tf.convert_to_tensor(y_pred)
    # y_true = tf.cast(y_true, y_pred.dtype)
    # print(type(y_true))
    # print(type(y_pred))
    # false_positives = tf.sum([y_t == 0 and y_p == 1 for y_t, y_p in zip(y_true, y_pred)])
    # false_negatives = tf.sum([y_t == 1 and y_p == 0 for y_t, y_p in zip(y_true, y_pred)])
    # print(type(false_positives))
    # print(type(y_pred))

    # precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    # recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    # return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    # # return f1_score(y_true, y_pred)
    # # metr = tf.keras.metrics.F1Score(threshold=0.5)
    # # metr.update_state(y_true, y_pred)
    # # return metr.result()


def tunas(y_true, y_pred, params):
    ac = acc(y_true, y_pred)
    beta = -0.4
    target_params = 100_000
    return ac + beta*tf.math.abs(params / target_params - 1)

def f1_tunas(y_true, y_pred, params):
    sc = f1(y_true, y_pred)
    beta = -0.4
    target_params = 10_000
    return sc + beta*tf.math.abs(params / target_params - 1)

def mae_tunas(y_true, y_pred, params):
    err = mae(y_true, y_pred)
    beta = -0.4
    target_params = 100_000
    return beta*tf.math.abs(params / target_params - 1) - err

def sparse_perplexity(y_true, y_pred):
    cross_entropy = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    perplexity = tf.pow(2.0, cross_entropy)
    return perplexity


def to_tfp(metric_func):
    """Convert a regular tensorflow-keras metric for tensorflow probability where the output is a distribution.

    Args:
        metric_func (func): A regular tensorflow-keras metric function.
    """

    @functools.wraps(metric_func)
    def wrapper(y_true, y_pred):
        return metric_func(y_true, y_pred.mean())

    wrapper.__name__ = f"tfp_{metric_func.__name__}"

    return wrapper


# convert some metrics for Tensorflow Probability where the output of the model is
# a distribution
tfp_r2 = to_tfp(r2)
tfp_mae = to_tfp(mae)
tfp_f1 = to_tfp(f1)
tfp_mse = to_tfp(mse)
tfp_rmse = to_tfp(rmse)

metrics_func = OrderedDict()
metrics_func["mean_absolute_error"] = metrics_func["mae"] = mae
metrics_func["r2"] = r2
metrics_func["f1"] = f1
metrics_func["mean_squared_error"] = metrics_func["mse"] = mse
metrics_func["root_mean_squared_error"] = metrics_func["rmse"] = rmse
metrics_func["accuracy"] = metrics_func["acc"] = acc
metrics_func["sparse_perplexity"] = sparse_perplexity

metrics_func["tunas"] = metrics_func["tunas_obj"] = tunas
metrics_func["mae_tunas"] = metrics_func["mae_tunas_obj"] = mae_tunas

metrics_func["tfp_r2"] = tfp_r2
metrics_func["tfp_mse"] = tfp_mse
metrics_func["tfp_mae"] = tfp_mae
metrics_func["tfp_f1"] = tfp_f1
metrics_func["tfp_rmse"] = tfp_rmse

metrics_obj = OrderedDict()
metrics_obj["auroc"] = lambda: tf.keras.metrics.AUC(name="auroc", curve="ROC")
metrics_obj["aucpr"] = lambda: tf.keras.metrics.AUC(name="aucpr", curve="PR")


def selectMetric(name: str):
    """Return the metric defined by name.

    Args:
        name (str): a string referenced in DeepHyper, one referenced in keras or an attribute name to import.

    Returns:
        str or callable: a string suppossing it is referenced in the keras framework or a callable taking (y_true, y_pred) as inputs and returning a tensor.
    """
    if callable(name):
        return name
    if metrics_func.get(name) is None and metrics_obj.get(name) is None:
        try:
            return load_attr(name)
        except Exception:
            return name  # supposing it is referenced in keras metrics
    else:
        if name in metrics_func:
            return metrics_func[name]
        else:
            return metrics_obj[name]()
