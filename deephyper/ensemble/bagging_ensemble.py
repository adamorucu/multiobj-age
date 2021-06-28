import argparse
import os
import traceback

import tensorflow as tf
import numpy as np
import ray

from deephyper.nas.metrics import selectMetric
from deephyper.core.parser import add_arguments_from_signature
from deephyper.ensemble import BaseEnsemble
from deephyper.nas.run.util import set_memory_growth_for_visible_gpus


def mse(y_true, y_pred):
    return tf.square(y_true - y_pred)


@ray.remote(num_cpus=1)
def model_predict(model_path, X):

    # GPU Configuration if available
    set_memory_growth_for_visible_gpus(True)
    tf.keras.backend.clear_session()
    model_file = model_path.split("/")[-1]

    try:
        print(f"Loading model {model_file}", end="\n", flush=True)
        model = tf.keras.models.load_model(model_path, compile=False)
    except:
        print(f"Could not load model {model_file}", flush=True)
        traceback.print_exc()
        model = None

    if model:
        y = model.predict(X, batch_size=1)
    else:
        y = None

    return y


class BaggingEnsembleRegressor(BaseEnsemble):
    def __init__(
        self,
        model_dir,
        loss=mse,
        size=5,
        verbose=True,
        ray_address="",
        num_cpus=1,
        num_gpus=None,
        selection="topk",
    ):
        super().__init__(
            model_dir,
            loss,
            size,
            verbose,
            ray_address,
            num_cpus,
            num_gpus,
        )
        self.selection = selection

    @staticmethod
    def _extend_parser(parser) -> argparse.ArgumentParser:
        add_arguments_from_signature(parser, BaggingEnsembleRegressor)
        return parser

    def fit(self, X, y):
        X_id = ray.put(X)

        model_files = self.list_all_model_files()
        model_path = lambda f: os.path.join(self.model_dir, f)

        y_pred = ray.get(
            [
                model_predict.options(
                    num_cpus=self.num_cpus, num_gpus=self.num_gpus
                ).remote(model_path(f), X_id)
                for f in model_files
            ]
        )
        y_pred = np.array([arr for arr in y_pred if arr is not None])

        members_indexes = topk(self.loss, y_true=y, y_pred=y_pred, k=self.size)
        self.members_files = [model_files[i] for i in members_indexes]

    def predict(self, X) -> np.ndarray:
        # make predictions
        X_id = ray.put(X)
        model_path = lambda f: os.path.join(self.model_dir, f)

        y_pred = ray.get(
            [
                model_predict.options(
                    num_cpus=self.num_cpus, num_gpus=self.num_gpus
                ).remote(model_path(f), X_id)
                for f in self.members_files
            ]
        )
        y_pred = np.array([arr for arr in y_pred if arr is not None])

        y = aggregate_predictions(y_pred, regression=True)

        return y

    def evaluate(self, X, y, metrics=None):
        scores = {}

        y_pred = self.predict(X)

        scores["loss"] = tf.reduce_mean(self.loss(y, y_pred)).numpy()
        # scores["loss"] = self.loss(y, y_pred).numpy()
        if metrics:
            for metric_name in metrics:
                scores[metric_name] = apply_metric(metric_name, y, y_pred)

        return scores


def apply_metric(metric_name, y_true, y_pred) -> float:
    metric_func = selectMetric(metric_name)
    metric = tf.reduce_mean(
        metric_func(
            tf.convert_to_tensor(y_true, dtype=np.float64),
            tf.convert_to_tensor(y_pred, dtype=np.float64),
        )
    ).numpy()
    return metric


def aggregate_predictions(y_pred, regression=True):
    """Build an ensemble from predictions.

    Args:
        ensemble_members (np.array): Indexes of selected members in the axis-0 of y_pred.
        y_pred (np.array): Predictions array of shape (n_models, n_samples, n_outputs).
        regression (bool): Boolean (True) if it is a regression (False) if it is a classification.
    Return:
        A TFP Normal Distribution in the case of regression and a np.array with average probabilities
        in the case of classification.
    """
    n = np.shape(y_pred)[0]
    print("n: ", n)
    y_pred = np.sum(y_pred, axis=0)
    if regression:
        agg_y_pred = y_pred / n
    else:  # classification
        agg_y_pred = np.argmax(y_pred, axis=1)
    return agg_y_pred


def topk(loss_func, y_true, y_pred, k=2):
    """Select the top-k models to be part of the ensemble. A model can appear only once in the ensemble for this strategy."""
    # losses is of shape: (n_models, n_outputs)
    print(np.shape(y_true), np.shape(y_pred))
    losses = tf.reduce_mean(loss_func(y_true, y_pred), axis=1).numpy()
    print(np.shape(losses))
    ensemble_members = np.argsort(losses, axis=0)[:k].reshape(-1).tolist()
    print("losses: ", losses[np.argsort(losses, axis=0)[:5]].flatten().tolist())
    print("ensemble_members: ", ensemble_members)
    return ensemble_members