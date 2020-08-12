from typing import Union, Callable, List, Any

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
import tensorflow as tf

MAX_SEQUENCE_LENGTH = 128


class SmilePredictor(BaseEstimator, ClassifierMixin):
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
                 batch_size: int = 32, loss: str = 'binary_crossentropy',
                 validation_split: float = 0.15, epochs: int = 100,
                 metrics: List[Union[str, Callable, None]] = None,
                 max_length: int = 128, class_weight=None
                 ):
        self.activation = activation
        self.optimizer = optimizer
        self.class_weight = class_weight
        self.validation_split = validation_split
        self.loss = loss
        self.epochs = epochs
        self.metrics = metrics
        self.batch_size = batch_size
        self.max_length = max_length

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
        # Create vocabulary with training smiles
        tk = tf.keras.preprocessing.text.Tokenizer(num_words=None,
                                                   char_level=True,
                                                   lower=False,
                                                   oov_token='UNK')
        tk.fit_on_texts(X)
        self.tk = tk

        # Vectorize training smiles.
        X_seq = tk.texts_to_sequences(X)

        # Get max sequence length.
        self.max_length = len(max(X, key=len))
        if self.max_length > MAX_SEQUENCE_LENGTH:
            self.max_length = MAX_SEQUENCE_LENGTH

        # Fix sequence length to max value. Sequences shorter than the length
        # are padded in the beginning and sequences longer are truncated
        # at the beginning.
        X_pad = tf.keras.preprocessing.sequence.pad_sequences(X_seq,
                                                              maxlen=self.max_length,
                                                              padding='post',
                                                              truncating='post')
        X_pad_reshaped = X_pad.reshape(X_pad.shape[0], X_pad.shape[1], 1)

        model = build_model(input_shape=X_pad_reshaped.shape[1:],
                            activation=self.activation)

        model.compile(loss=self.loss,
                      optimizer=self.optimizer,
                      metrics=self.metrics)
        model.summary()

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            verbose=1,
            patience=20,
            mode='max',
            restore_best_weights=True)

        self.history = model.fit(
            X_pad_reshaped,
            y,
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
        Arguments:
            X : array-like, shape `(n_samples, n_features)`
                Test samples where `n_samples` is the number of samples
                and `n_features` is the number of features.

        Returns:
            preds: array-like, shape `(n_samples,)`
                Class predictions.
        """
        check_is_fitted(self, 'history')
        X = pd.Series(X)

        # Vectorize testing smiles.
        X_seq = self.tk.texts_to_sequences(X)
        X_pad = tf.keras.preprocessing.sequence.pad_sequences(X_seq,
                                                              maxlen=self.max_length,
                                                              padding='post',
                                                              truncating='post')
        X_pad_reshaped = X_pad.reshape(X_pad.shape[0], X_pad.shape[1], 1)

        probs = self.history.model.predict(X_pad_reshaped.astype('float64'))
        preds = np.array(probs > 0.5).astype('int')
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


def build_model(input_shape, activation):
    """
    Builds a CNN model

    """

    input_layer = tf.keras.layers.Input(shape=input_shape)

    conv1 = tf.keras.layers.Conv1D(filters=64, kernel_size=5, padding='same')(
        input_layer)
    conv1 = tf.keras.layers.BatchNormalization()(conv1)
    conv1 = tf.keras.layers.Activation(activation=activation)(conv1)

    conv2 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same')(
        conv1)
    conv2 = tf.keras.layers.BatchNormalization()(conv2)
    conv2 = tf.keras.layers.Activation('relu')(conv2)

    gap_layer = tf.keras.layers.GlobalAveragePooling1D()(conv2)

    # dropout = tf.keras.layers.Dropout(rate=0.3)(gap_layer)
    output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(
        gap_layer)

    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

    return model
