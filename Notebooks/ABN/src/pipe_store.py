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
def data_loader(path:str, **kwargs):
    return pd.read_csv(path, sep=';', **kwargs)

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

@logging
def one_hot_encoder(df:pd.DataFrame, col, *cats):
    """
    Convert categorical variable into dummy variables and keeps the given categorical columns. 
    Under the hood uses pandas get_dummies method

    Parameters:
    -----------
    col: The specific column to be one hot encoded
    
    cats: the categories/columns that will be kept after one hot encoding.

    Example:
    --------
    >> s = pd.Series(list('abca'))
    >> pd.get_dummies(s)
       a  b  c
    0  1  0  0
    1  0  1  0
    2  0  0  1
    3  1  0  0
    """
    
    encoded_df = pd.get_dummies(df[col])    
    df.drop(col, axis=1, inplace=True)
    if not cats:
        return pd.concat([df, encoded_df], axis=1)
    cats = list(cats)    
    return pd.concat([df, encoded_df[cats]], axis=1)

@logging
def integer_encoder(df:pd.DataFrame, *cols):
    """
    Encode the object as an enumerated type, i.e. Integer Encoding. This function is particularly 
    used for features with 2 category. Under the hood uses pandas factorize method. For Features 
    with more categories consider one_hot_encoder.

    Parameters:
    -----------
    cols: Name of the column/feature in the dataset
    
    Example:
    --------
    >> pd.factorize(['b', 'b', 'a', 'c', 'b'])
    (
        array([0, 0, 1, 2, 0]...),
        array(['b', 'a', 'c'], dtype=object)
    )
    """

    if not cols:
        raise ValueError('Name of the dataset column is missing')

    for col in cols:
        if df[col].dtype not in ['object', 'category']:
            raise TypeError(f'{col} dtype not suitable for encoding')
        df[col], _ = pd.factorize(df[col])
    return df


def sklearn_adapter(df:pd.DataFrame, label:str):
    """ Splits the data frame to input matrix and output vector suitable for sklearn package adapter """
    _df = df.copy(deep=True)
    return (_df.drop(label, axis=1), _df.pop(label))
