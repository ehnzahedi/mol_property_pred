from typing import Union, Callable, List, Any

import numpy as np
from pathlib import Path
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
import tensorflow as tf
from tensorflow.keras.utils import to_categorical


class FeatureExtractedClassifier(BaseEstimator, ClassifierMixin):
    """
    An estimator that takes the extracted features of a molecule as input
    and predicts the P1 property.

    Parameters
    ----------
    activation : Activation function to use.
        If you don't specify anything, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
    batch_size : Integer or `None`.
        Number of samples per gradient update.
        If unspecified, `batch_size` will default to 32.
    epochs : Integer. Number of epochs to train the model.
        An epoch is an iteration over the entire `x` and `y`
        data provided.
    validation_split: Float between 0 and 1.
        Fraction of the training data to be used as validation data.
        The model will set apart this fraction of the training data,
        will not train on it, and will evaluate the loss and any model metrics
        on this data at the end of each epoch.
    class_weight : Optional dictionary mapping class indices (integers)
        to a weight (float) value, used for weighting the loss function
        (during training only).
        This can be useful to tell the model to "pay more attention" to
        samples from an under-represented class.
    optimizer : String (name of optimizer) or optimizer instance.
        See `tf.keras.optimizers`.
    loss : String (name of objective function), objective function or
        `tf.losses.Loss` instance. See `tf.losses`.
    metrics : List of metrics to be evaluated by the model during training
        and testing. Typically you will use `metrics=['accuracy']`.


    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The classes labels.
    n_classes_ : int
        The number of class lebels.
    history : object
        a dictionary recording training loss values and metrics values at
        successive epochs, as well as validation loss values and validation
        metrics values (if applicable).
    """

    def __init__(self, activation: str = 'relu',
                 optimizer: Any = 'adam',
                 batch_size: int = 32, loss: str = 'categorical_crossentropy',
                 validation_split: float = 0.2, epochs: int = 100,
                 metrics: List[Union[str, Callable, None]] = None,
                 class_weight=None
                 ):
        self.activation = activation
        self.optimizer = optimizer
        self.class_weight = class_weight
        self.validation_split = validation_split
        self.loss = loss
        self.epochs = epochs
        self.metrics = metrics
        self.batch_size = batch_size

    def fit(self, X, y):
        """ Fits the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like of shape (n_samples,)
            Target vector relative to X.

        Returns
        -------
        self
            Fitted estimator.
        """

        if len(y.shape) == 2 and y.shape[1] > 1:
            self.classes_ = np.arange(y.shape[1])
        elif (len(y.shape) == 2 and y.shape[1] == 1) or len(y.shape) == 1:
            y = to_categorical(y)

        else:
            raise ValueError('Invalid shape for y: ' + str(y.shape))
        self.n_classes_ = len(np.unique(y))

        model = build_model(input_shape=X.shape[1:],
                            activation=self.activation,
                            n_outputs=self.n_classes_,
                            optimizer=self.optimizer,
                            loss=self.loss,
                            metrics=self.metrics)

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            verbose=0,
            patience=25,
            mode='max',
            restore_best_weights=True)

        self.history = model.fit(X, y,
                                 batch_size=self.batch_size,
                                 epochs=self.epochs,
                                 callbacks=[early_stopping],
                                 validation_split=self.validation_split,
                                 class_weight=self.class_weight,
                                 verbose=2
                                 )
        return self

    def predict(self, X):
        """Returns the class predictions for the given test data.

        Parameters
        ----------
            X : array-like, shape `(n_samples, n_features)`
                Test samples where `n_samples` is the number of samples
                and `n_features` is the number of features.

        Returns
        ------
            preds : array-like, shape `(n_samples,)`
                Class predictions.
        """
        check_is_fitted(self, 'history')
        probs = self.history.model.predict(X)
        preds = np.argmax(probs, axis=1)
        return preds

    def save_model(self):
        parent_dir = Path().resolve().parent.absolute()
        save_path = parent_dir / 'saved_models'
        self.history.model.save("saved_models/model1.h5")

    def load_model(self):
        reconstructed_model = tf.keras.models.load_model(
            "saved_models/model1.h5")
        reconstructed_model.compile(optimizer=self.optimizer,
                                    loss=self.loss,
                                    metrics=self.metrics)
        return reconstructed_model


def build_model(input_shape, activation, n_outputs, optimizer, loss, metrics):
    """
    Builds a DNN model

    """

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(256, input_shape=input_shape,
                              activation=activation),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation=activation),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation=activation),
        tf.keras.layers.Dense(16, activation=activation),
        tf.keras.layers.BatchNormalization(axis=1),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(n_outputs, activation="softmax")
    ])

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)

    return model
