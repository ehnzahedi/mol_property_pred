import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import hamming_loss
import tensorflow as tf
from feature_extractor import fingerprint_features

from classifiers.model1 import FeatureExtractedClassifier
# from classifiers.model2 import SmilePredictor
from get_data import fetch_single_label_data, fetch_multi_label_data
from utils import plot_cm, plot_metrics

# we use class weight since the data is imbalanced
# the following class weights is calculated in the data_exploration file
CLASS_WEIGHTS = {0: 2.81, 1: 0.61}

# metrics for the evaluation of DL models
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


def train_model3(X, y, metrics):
    clf = FeatureExtractedClassifier(epochs=2,
                                     batch_size=16,
                                     metrics=metrics,
                                     loss='binary_crossentropy',
                                     output_activation='sigmoid',
                                     optimizer=tf.keras.optimizers.Adam(
                                         lr=1e-5),
                                     is_multi_label=True
                                     )

    clf.fit(X, y)
    plot_metrics(clf.history)
    print("model3 trained!")
    return clf


def evaluate_model(clf, X, y):
    probs = clf.predict(X)
    test_pred = np.argmax(probs, axis=1)
    plot_cm(y, test_pred)
    plt.show()


def evaluate_model3(clf, X, y):
    probs = clf.predict(X)
    preds = [prob > 0.5 for prob in probs]
    test_pred = np.array(preds).astype('int')
    plot_cm(y, test_pred)
    plt.show()
    hamming_loss_metric = hamming_loss(y_test, test_pred)
    print(f"hamming loss: {hamming_loss_metric:.4f}")


def predict_smile(clf, s):
    pred = clf.predict(s)
    label = np.argmax(pred, axis=1)[0]
    print(f"smile property prediction: P1={label}")


def predict_smile3(clf, s):
    probs = clf.predict(s)
    preds = [prob > 0.5 for prob in probs]
    label_pred = np.array(preds).astype('int')
    print(f"smile property prediction: P1={label_pred}")


if __name__ == "__main__":

    args = sys.argv

    if len(args) > 1:
        if sys.argv[1] == 'model1':
            X_train, X_test, y_train, y_test = fetch_single_label_data(
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
                print(f"smile: {smile}")
                x = preprocess_smile(str(smile))
                predict_smile(clf1, x)

        if sys.argv[1] == 'model3':
            X_train, X_test, y_train, y_test = fetch_multi_label_data(
                get_extracted_features=True)
            if sys.argv[2] == 'train':
                clf1 = train_model3(X_train, y_train, metrics=METRICS)
                clf1.history.model.save("saved_models/model3.h5")
                print("model3 saved!")
            if sys.argv[2] == 'evaluate':
                print("evaluation...")
                clf1 = tf.keras.models.load_model("saved_models/model3.h5")
                evaluate_model3(clf1, X_test, y_test)
            if sys.argv[2] == 'predict':
                smile = sys.argv[3]
                clf1 = tf.keras.models.load_model("saved_models/model3.h5")
                print(f"smile: {smile}")
                x = preprocess_smile(str(smile))
                predict_smile3(clf1, x)
    else:
        print("You need to enter your desired model, "
              "[train, evaluate, predict]")