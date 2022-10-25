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
    no_skill = sum(y_test) / len(y_test) # No skill predicts all as positive class, precision = (y_test=1)[TP] / (y_test=1)[TP] + (y_test=0)[FP]
    plot_obj.plot([0, 1], [no_skill, no_skill], linestyle='--')
    plot_obj.plot(recall, precision, label=f'PR curve (area = {auc(recall, precision):0.2f})')
    xlabel, ylabel = "Recall", "Precision"
    ax.set_xlabel(xlabel) if ax else plt.xlabel(xlabel)
    ax.set_ylabel(ylabel) if ax else plt.xlabel(ylabel)
    title = "Precision Recall Curve"
    ax.set_title(title) if ax else plt.title(title)
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


def print_concordance_index(estimator, X_test, T='churn_time', E='churn_event'):
    from lifelines.utils import concordance_index
    c_index = concordance_index(
        X_test[T], 
        -estimator.predict_partial_hazard(X_test), 
        X_test[E]
    )
    print(f'C-index for test data is{c_index * 100: .2f}%')


def t2e_model_performance(estimator, X_test):
    import seaborn as sns
    inds_churn = X_test.churn_event == 1
    inds_no_churn = X_test.churn_event == 0
    pred_hazard = estimator.predict_cumulative_hazard(X_test).T
    fig, ax = plt.subplots(1, 2, figsize=(14,4))
    sns.histplot(pred_hazard[inds_churn][24.0], stat='percent', ax=ax[0], bins=[0, .1, .2, .3,  2.5])
    ax[0].set_title(f'#Clients that Chruned {len(pred_hazard[inds_churn])}, Recall=74%, Precision=50%')
    ax[0].set_xlabel('Max Cumulative Hazard of churn (cut-off = 0.3)')
    sns.histplot(pred_hazard[inds_no_churn][24.0], stat='percent', ax=ax[1], bins=[0, .1, .2, .3,  2.5])
    ax[1].set_title(f'#Clients that Not Chruned {len(pred_hazard[inds_no_churn])}')
    ax[1].set_xlabel('Max Cumulative Hazard of churn (cut-off = 0.3)');

def plot_cumulative_hazard_and_survival(estimator, X_test, cut_off):
    inds_churn = X_test.churn_event == 1
    ind_churn_sam = X_test[inds_churn].sample().index
    inds_no_churn = X_test.churn_event == 0
    ind_no_churn_sam = X_test[inds_no_churn].sample().index
    
    pred_hazard = estimator.predict_cumulative_hazard(X_test).T
    pred_surv = estimator.predict_survival_function(X_test).T
    fig, ax = plt.subplots(1,2, figsize=(12,4))

    pred_surv.loc[ind_no_churn_sam, :].T.plot(ax=ax[0])
    pred_surv.loc[ind_churn_sam, :].T.plot(ax=ax[0])
    ax[0].legend(['Not Churned client', 'Churned client'])
    ax[0].set_xlabel('Time(Month)')
    ax[0].set_ylabel('Probability of Retention');

    pred_hazard.loc[ind_no_churn_sam, :].T.plot(ax=ax[1])
    pred_hazard.loc[ind_churn_sam, :].T.plot(ax=ax[1])
    ax[1].legend(['Not Churned client', 'Churned client'])
    ax[1].plot([0, 25], [cut_off, cut_off], linestyle='--')
    ax[1].set_xlabel('Time(Month)')
    ax[1].set_ylabel('Cumulative Hazard of churn');