import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns


def plot_metrics(history, metrics=None):
    """
    Plots the metrics of deep learning models for training and validation
    """
    if metrics is None:
        metrics = ['loss', 'auc', 'accuracy', 'recall']
    plt.figure()
    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        plt.subplot(2, 2, n + 1)
        plt.plot(history.epoch, history.history[metric], label='Train')
        plt.plot(history.epoch, history.history['val_' + metric],
                 linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        plt.legend()


def plot_cm(labels, predictions):
    """
    Plots the confusion matrix and prints classification report
    """
    if len(labels.shape) == 2 and labels.shape[1] > 1:
        labels = np.argmax(labels, axis=1)
    if len(predictions.shape) == 2 and predictions.shape[1] > 1:
        predictions = np.argmax(predictions, axis=1)

    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    print(classification_report(labels, predictions))


