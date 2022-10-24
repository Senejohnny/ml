

from sklearn.model_selection import cross_validate, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
random_state = 42

regs = [ LogisticRegression(), RandomForestClassifier()]


results = {}
for reg in regs:

    pipeline = Pipeline([
        ('Scaling', column_transformer_scaler),
        ('Model', reg),
    ], verbose=False)

    kfs = KFold(n_splits=5, shuffle=True, random_state=random_state)
    # For the list of all metrics visit: https://scikit-learn.org/stable/modules/model_evaluation.html
    metrics = ['recall', 'roc_auc', 'accuracy', 'f1'] 
    scores = cross_validate(pipeline, X, y, cv=kfs, scoring=metrics)
    # We will not use cross_val_score as it can only accept one metric
    # print(scores)
    reg_name = type(reg).__name__
    results[reg_name] = {key: round(np.mean(val), 3) for key, val in scores.items()}
pd.DataFrame(results)