import matplotlib.pyplot as plt

def plot_churn_risk(df, col, estimator, T:str='churn_time', E:str='event',  horizon=12):
    """ Plot univariate churn risk """
    if col not in df.columns:
        raise KeyError(f'{col} not in data frame')
    fig, ax = plt.subplots(figsize=(6, 5))
    timeline = np.linspace(0, horizon, 1000)
    vals = set(df[col].values)
    estimators = [estimator(label=col+'_'+str(val)) for val in vals]
    inds = [df_red[col] == val for val in vals]
    estimators = [estimator.fit(df_red[inds[i]][T], df_red[inds[i]][E], timeline=timeline) 
                    for i, estimator in enumerate(estimators)]
    # if type(estimator) in ['NelsonAalenFitter', 'WeibullFitter']:
    [estimator.plot(ax=ax) for estimator in estimators]
    add_at_risk_counts(*estimators, ax=ax, fontsize=10, rows_to_show=['At risk','Events'])
    ax.set_ylim(0.7)
    ax.set_xlabel('Months', fontsize=10)
    ax.set_ylabel('Risk of churn', fontsize=10)
