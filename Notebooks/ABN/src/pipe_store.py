""" This is a store of all the relevant functions consumed by pandas pipeline """
import logging
from datetime import datetime 
from functools import wraps
import pandas as pd 

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def logging(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        tic = datetime.now()
        result = func(*args, **kwargs)
        time_taken = str((datetime.now() - tic).total_seconds())
        logger.info(f"Step: {func.__name__} | Shape: {result.shape} | Computation Time: {time_taken}s")
        return result
    return wrapper

@logging
def set_data_types(df:pd.DataFrame, data_type:dict) -> pd.DataFrame:
    """ Setting data types to a  """
    for col in data_type.keys():
        if col not in df.columns:
            raise KeyError(f'Column {col} not exist')

    for col, dtype in data_type.items():
        try:
            df[col] = df[col].astype(dtype)
        except:
            print(f'Column {col} encounters error')
            raise
    return df


def sklearn_adapter(df:pd.DataFrame, label:str):
    """ Splits the data frame to input matrix and output vector suitable for sklearn package adapter """
    _df = df.copy(deep=True)
    return (_df.drop(label, axis=1), _df.pop(label))



def plot_correlations(df, verbose=True, **kwargs):
    """ If verbose variables with correlation above 0.6 are printed """
    
    # Correlation matrix for numerical variables
    corr_mat = df.corr()
    plt.figure(figsize=(10, 6))
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
        print('Top Correlations above 0.6')
        print(top_corr[top_corr > 0.6])

