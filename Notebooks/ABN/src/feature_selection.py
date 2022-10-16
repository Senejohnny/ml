import pandas as pd
import numpy as np
from seaborn import heatmap
import matplotlib.pyplot as plt

def plot_correlations(df:pd.DataFrame, figsize:tuple=(14, 8), verbose:bool=True, **kwargs) -> None:
    """
    Compute correlation of all variables X and y.
    If verbose variables with correlation above 0.6 are printed
    """
    
    # Correlation matrix for numerical variables
    corr_mat = df.corr()
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
        corr_threshold = 0.8
        print(f'Top Correlations above {corr_threshold}')
        print(top_corr[top_corr > corr_threshold])
        return top_corr[top_corr > corr_threshold]


def plot_corr_cat(df:pd.DataFrame, figsize:tuple=(7, 5), **kwargs):
    cols = df.columns
    import itertools
    from sklearn.feature_selection import chi2
    p_vals = [chi2(df[var1].values.reshape(-1, 1), df[var2].values.reshape(-1, 1))[1]
            for var1, var2 in list(itertools.product(cols, cols))]
    corr_mat = pd.DataFrame(np.array(p_vals).reshape(3, 3), columns=cols, index=cols)
    plt.figure(figsize=figsize)
    heatmap(corr_mat, cmap='Blues', annot=True, **kwargs)
    plt.show()

