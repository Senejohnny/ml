import pandas as pd
import numpy as np
from seaborn import heatmap
import matplotlib.pyplot as plt


def date_parser_o(x):
    refrence_month = pd.to_datetime('2011-1-1').to_period('M')
    trans_month = pd.to_datetime(x, format='%Y%m').to_period('M')
    return ( trans_month - refrence_month).n #n captures the month number

date_parser = lambda x: pd.to_datetime(x, format='%Y%m')

def plot_correlations(df:pd.DataFrame, show_figure:bool=True, figsize:tuple=(14, 8), verbose:bool=True, **kwargs) -> None:
    """
    Compute correlation of all variables X and y.
    If verbose variables with correlation above 0.6 are printed
    """
    
    # Correlation matrix for numerical variables
    corr_mat = df.corr()
    if show_figure:
        plt.figure(figsize=figsize)
        heatmap(corr_mat, cmap='RdYlGn', annot=True, **kwargs)
        plt.show()

    if verbose:
        top_corr = (
            corr_mat
            .abs()
            .where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
            .stack()
            .sort_values(ascending=False)
        )
        corr_threshold = 0.70
        print(f'Top Correlations above {corr_threshold}')
        return top_corr[top_corr > corr_threshold]


def plot_corr_cat(df:pd.DataFrame, show_figure:bool=True, figsize:tuple=(7, 5), verbose=True, **kwargs):
    # https://pythonfordatascienceorg.wordpress.com/chi-square-python/
    cols = df.columns
    size = len(cols)
    import itertools
    from sklearn.feature_selection import chi2
    p_vals = [chi2(df[var1].values.reshape(-1, 1), df[var2].values.reshape(-1, 1))[1]
            for var1, var2 in list(itertools.product(cols, cols))]
    corr_mat = pd.DataFrame(np.array(p_vals).reshape(size, size), columns=cols, index=cols)
    if show_figure:
        plt.figure(figsize=figsize)
        heatmap(corr_mat, cmap='Blues', annot=True, **kwargs)
        plt.show()
    if verbose:
        top_corr = (
            corr_mat
            .abs()
            .where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
            .stack()
            .sort_values(ascending=False)
        )
        corr_threshold = 0.05
        print(f'Correlations with signigicance < {corr_threshold}')
        return top_corr[top_corr < corr_threshold]


def total_variable_variances(df, top:int=5, include_label:bool=False):
    if not include_label:
        varainces = df.drop(['COMMERCIALLY_CHURNED', 'CHURNED_IND'], axis=1).var(numeric_only=True).round(2)
    else:
        varainces = df.var(numeric_only=True).round(2)
    print('Highest variances\n', dict(varainces.sort_values(ascending=False)[0:top]))
    print('Lowest variances\n', dict(varainces.sort_values(ascending=True)[0:top]))


def variable_variances_per_client(df):
    variances = {}
    for cust_id in df.CUSTOMER_ID.unique():
        df_id = df[df.CUSTOMER_ID == cust_id]
        variances[cust_id] = dict(df_id.var(numeric_only=True).round(2))
    return pd.DataFrame(variances).T

