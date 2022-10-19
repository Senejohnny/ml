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



def date_parser_o(x):
    refrence_month = pd.to_datetime('2011-1-1').to_period('M')
    trans_month = pd.to_datetime(x, format='%Y%m').to_period('M')
    return ( trans_month - refrence_month).n #n captures the month number

def date_parser(date_time:str):
    return pd.to_datetime(date_time, format='%Y%m')

@logging
def datetime2int(df:pd.DataFrame, time_col:str) -> pd.DataFrame():
    """ sorts the values per client id by certain column """
    date2month_numb = lambda x: (x.to_period('M') - pd.to_datetime('2011-1-1').to_period('M')).n
    df[time_col] = df[time_col].apply(date2month_numb).astype('float16')
    return df


@logging
def sort_values_per_client(df:pd.DataFrame, by:str) -> pd.DataFrame():
    """ sorts the values per client id by certain column """

    for client_id in df.CUSTOMER_ID.unique():
        ind = df.CUSTOMER_ID.eq(client_id)
        df[ind] = df[ind].sort_values(by='MONTH_PERIOD', ascending=False)
    return df

@logging
def data_loader(path:str, **kwargs,):
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
def clean_string_strip(df, *cols):
    for col in list(cols):
        df[col] = df[col].apply(str.strip)
    return df

@logging
def one_hot_encoder(df:pd.DataFrame, *col, **kwargs):
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
    df_org = df.copy(deep=True)
    cols = list(col)
    return pd.get_dummies(df_org, columns=cols, **kwargs) 


@logging
def integer_encoder(df:pd.DataFrame, *cols) -> pd.DataFrame():
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
        df[col], _ = pd.factorize(df[col], use_na_sentinel=False)
    return df

    


def sklearn_adapter(df:pd.DataFrame, label:str) -> tuple:
    """ Splits the data frame to input matrix and output vector suitable for sklearn package adapter """
    _df = df.copy(deep=True)
    return (_df.drop(label, axis=1), _df.pop(label))
