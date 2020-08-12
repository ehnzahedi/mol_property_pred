import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from feature_extractor import fingerprint_features

from classifiers.model1 import FeatureExtractedClassifier
from classifiers.model2 import SmilePredictor
from get_data import fetch_single_class_data
from utils import plot_cm, plot_metrics

CLASS_WEIGHTS = {0: 2.81, 1: 0.61}

METRICS = [
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc')
]


def preprocess_smile(s):
    s = str(s)
    extracted_features = np.array(fingerprint_features(s, size=2048))
    extracted_features = extracted_features.reshape(1, -1)
    return extracted_features


def train_model1(X, y, metrics, class_weight):
    clf = FeatureExtractedClassifier(epochs=2,
                                     batch_size=16,
                                     metrics=metrics,
                                     optimizer=tf.keras.optimizers.Adam(
                                         lr=1e-5),
                                     class_weight=class_weight
                                     )
    clf.fit(X, y)
    plot_metrics(clf.history)
    print("model1 trained!")
    return clf


def train_model2(X, y, metrics, class_weight):
    clf = SmilePredictor(epochs=2,
                         batch_size=16,
                         loss='binary_crossentropy',
                         optimizer=tf.keras.optimizers.SGD(
                             learning_rate=0.001,
                             momentum=0.0,
                             nesterov=False),
                         metrics=metrics,
                         class_weight=class_weight,
                         validation_split=0.2
                         )

    clf.fit(X, y)
    plot_metrics(clf.history)
    print("model2 trained!")
    return clf


def evaluate_model(clf, X, y):
    test_pred = clf.predict(X)
    plot_cm(y, test_pred)
    plt.show()


def predict_smile(clf, s):
    pred = clf.predict(s)
    label = np.argmax(pred, axis=1)[0]
    print(f"smile: {s} \nsmile property prediction: P1={label}")


if __name__ == "__main__":
    if sys.argv[1] == 'model1':
        X_train, X_test, y_train, y_test = fetch_single_class_data(
            get_extracted_features=True)
        if sys.argv[2] == 'train':
            clf1 = train_model1(X_train, y_train, metrics=METRICS,
                                class_weight=CLASS_WEIGHTS)
            clf1.history.model.save("saved_models/model1.h5")
            print("model1 saved!")
        if sys.argv[2] == 'evaluate':
            print("evaluation...")
            clf1 = tf.keras.models.load_model("saved_models/model1.h5")
            evaluate_model(clf1, X_test, y_test)
        if sys.argv[2] == 'predict':
            smile = sys.argv[3]
            clf1 = tf.keras.models.load_model("saved_models/model1.h5")
            x = preprocess_smile(str(smile))
            predict_smile(clf1, x)
