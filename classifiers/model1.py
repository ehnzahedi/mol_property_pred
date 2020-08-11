from typing import Union, Callable, List

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


    """

    def __init__(self, activation: str = 'relu',
                 optimizer: Union[str, Callable] = 'adam',
                 batch_size: int = 32, loss='categorical_crossentropy',
                 validation_split: float = 0.15, epochs: int = 100,
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

    def fit(self, X, y, **kwargs):
        """
        Fit the model according to the given training data.
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
            patience=20,
            mode='max',
            restore_best_weights=True)

        self.history = model.fit(X, y,
                                 batch_size=self.batch_size,
                                 epochs=self.epochs,
                                 callbacks=[early_stopping],
                                 validation_split=0.2
                                 )
        return self

    def predict(self, X):
        """Returns the class predictions for the given test data.
        Arguments:
            X : array-like, shape `(n_samples, n_features)`
                Test samples where `n_samples` is the number of samples
                and `n_features` is the number of features.

        Returns:
            preds: array-like, shape `(n_samples,)`
                Class predictions.
        """
        check_is_fitted(self, 'history')
        probs = self.history.model.predict(X)
        return np.argmax(probs, axis=1)

    def save_model(self):
        # saving model
        # json_model = self.history.model.to_json()
        # open('model_architecture.json', 'w').write(json_model)
        # # saving weights
        # self.history.model.save_weights('model_weights.h5', overwrite=True)

        # tf.keras.classifiers.save_model(self.history.model, 'saved_model1')
        parent_dir = Path().resolve().parent.absolute()
        save_path = parent_dir / 'saved_models'
        self.history.model.save("saved_models/model1.h5")

    def load_model(self):
        # loading model
        # model = tf.keras.classifiers.load_model('saved_model1')
        # model.compile(optimizer=self.optimizer,
        #               loss=self.loss,
        #               metrics=self.metrics)

        # loading model
        # model = tf.keras.classifiers.model_from_json(
        #     open('model_architecture.json').read())
        # model.load_weights('model_weights.h5')
        # model.compile(optimizer=self.optimizer,
        #               loss=self.loss,
        #               metrics=self.metrics)

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
