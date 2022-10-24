import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    auc,
    f1_score,
    roc_curve,
    roc_auc_score,
    precision_recall_curve, 
    ConfusionMatrixDisplay,
    recall_score,
    precision_score,
)

def plot_confusion_matrix(estimator, X_test, y_test, ax=None):
    """ 
    Plot confusion matrix given below parameters

    Parameters:
    ----------
    estimator: sklearn pipeline/model | keras model 
    X_test: dataframe or array-like of shape (n_samples, m_featuers)
    y_test: array-like of shape (n_samples,)
    """
    
    from copy import deepcopy
    X_test = deepcopy(X_test)

    labels = ['Not Churned', 'Churned']
    y_pred = estimator.predict(X_test)
    # Convert a possible continous prediction to binary. 
    y_pred = (y_pred.reshape(-1) >= 0.5).astype(int)
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=labels, ax=ax)
    plt.title("Confusion Matrix Client Churn")
    plt.show()

def plot_roc_curve(pipeline, X_test, y_test, ax=None):
    """ """
    plot_obj = ax if ax else plt
    X_test = X_test.copy()
    # predict probabilities
    clf_probs = pipeline.predict_proba(X_test)
    # keep probabilities for the positive outcome only
    clf_probs = clf_probs[:, 1]
    # calculate AUC score
    auc = roc_auc_score(y_test, clf_probs)
    # calculate roc curves
    fpr, tpr, _ = roc_curve(y_test, clf_probs)
    ns_fpr, ns_tpr, _ = roc_curve(y_test, np.zeros(len(y_test))) # This is a no skill classifier
    # plot the roc curve for the model
    plot_obj.plot(ns_fpr, ns_tpr, linestyle='--', label=f"ROC curve (area = {auc:0.2f})")
    plot_obj.plot(fpr, tpr)
    xlabel, ylabel = "False Positive Rate (1- Specificity)", "True Positive Rate (Sensitivity)"
    ax.set_xlabel(xlabel) if ax else plt.xlabel(xlabel)
    ax.set_ylabel(ylabel) if ax else plt.xlabel(ylabel)
    title = "Receiver Operating Characteristic (ROC)"
    ax.set_title(title) if ax else plt.title(title)
    plot_obj.legend(loc="lower right")
    # fig.show() if ax else plt.show()

def plot_precision_recall_curve(pipeline, X_test, y_test, ax=None):
    
    plot_obj = ax if ax else plt
    X_test = X_test.copy()
    # predict probabilities
    clf_probs = pipeline.predict_proba(X_test)
    # keep probabilities for the positive outcome only
    clf_probs = clf_probs[:, 1]
    # predict class values
    precision, recall, _ = precision_recall_curve(y_test, clf_probs)
    # plot the precision-recall curves
    no_skill = sum(y_test) / len(y_test) # a no skill predicts all to
    plot_obj.plot([0, 1], [no_skill, no_skill], linestyle='--')
    plot_obj.plot(recall, precision, label=f'PR curve (area = {auc(recall, precision):0.2f})')
    xlabel, ylabel = "Recall", "Precision"
    ax.set_xlabel(xlabel) if ax else plt.xlabel(xlabel)
    ax.set_ylabel(ylabel) if ax else plt.xlabel(ylabel)
    plot_obj.legend()
    # fig.show() if ax else plt.show()

def plot_calibration(estimator, X_test, y_test, **kwargs):
    CalibrationDisplay.from_estimator(estimator, X_test, y_test, **kwargs)
    plt.show()


def print_scores(estimator, X_test, y_test):
    y_pred = estimator.predict(X_test)
    print(f'Precision/PPV := TP/(TP+FP): {precision_score(y_test, y_pred): 0.2}')
    print(f'Sensitivity/Recall/TPR := TP/(TP+FN): {recall_score(y_test, y_pred): 0.2}')
    # print(f'Specifcity/TNR := TP/(TP+FP): ?')

