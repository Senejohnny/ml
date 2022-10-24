""" This is a store of all the relevant functions consumed by pandas pipeline """
import logging
from collections import defaultdict
from datetime import datetime 
from functools import wraps
from typing import Literal
import pandas as pd 
from sklearn.preprocessing import LabelEncoder

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
    
    df_t2e = df.copy(deep=True)
    date2month_numb = lambda x: (x.to_period('M') - pd.to_datetime('2011-1-1').to_period('M')).n
    df_t2e[time_col] = df_t2e[time_col].apply(date2month_numb).astype('float16')
    return df_t2e


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

# @logging
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
def label_encoder(df:pd.DataFrame, *cols) -> pd.DataFrame():
    """
    Encode the object as an enumerated type, i.e. Label Encoding. This function uses
    sklearn label encoder class under the hood

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
    
    df_enc = df.copy(deep=True) 
    for col in list(cols):
        encoder = LabelEncoder()
        df_enc[col] = encoder.fit_transform(df_enc[col])
    return df_enc

# @logging
# def integer_encoder(df:pd.DataFrame, *cols) -> pd.DataFrame():
#     """
#     Encode the object as an enumerated type, i.e. Integer Encoding. This function is particularly 
#     used for features with 2 category. Under the hood uses pandas factorize method. For Features 
#     with more categories consider one_hot_encoder.

#     Parameters:
#     -----------
#     cols: Name of the column/feature in the dataset
    
#     Example:
#     --------
#     >> pd.factorize(['b', 'b', 'a', 'c', 'b'])
#     (
#         array([0, 0, 1, 2, 0]...),
#         array(['b', 'a', 'c'], dtype=object)
#     )
#     """

#     if not cols:
#         raise ValueError('Name of the dataset column is missing')

#     for col in cols:
#         if df[col].dtype not in ['object', 'category']:
#             raise TypeError(f'{col} dtype not suitable for encoding')
#         df[col], _ = pd.factorize(df[col], use_na_sentinel=False)
#     return df

    
def sklearn_adapter(df:pd.DataFrame, label:str) -> tuple:
    """ Splits the data frame to input matrix and output vector suitable for sklearn package adapter """
    _df = df.copy(deep=True)
    return (_df.drop(label, axis=1), _df.pop(label))



def summerize_client_behaviour(df, churn_col:Literal['CHURNED_IND', 'COMMERCIALLY_CHURNED'], horizon:int=6):
    """ Extract client behaviours throughout time and summerise the data """
    _dics = defaultdict(lambda: defaultdict(dict))
    ids_rejoin = []
    ids_rechurn = []
    for id in df.CUSTOMER_ID.unique():
        try:    
            df_id = (
                df[df.CUSTOMER_ID.eq(id)]
                .sort_values(by='MONTH_PERIOD', ascending=False)
                .reset_index(drop=True)
            )
            # client switches active -> churn -> active 
            if df_id[churn_col].diff().abs().sum() > 1:
                # print(f'Client {id}, active -> churn -> active ')
                ids_rechurn.append(id)
                continue
            # client starts with churn status 
            if df_id[churn_col].values[-1]:
                # print(f'Client {id} churn -> active')
                ids_rejoin.append(id)
                continue

            if 1 in df_id[churn_col].values:
                _dics[id]['churn_event'] = 1
                ind_lt_1 = df_id[churn_col].lt(1)
                ind_churn = df_id[df_id[churn_col].lt(1)].index.min()
                _dics[id]['churn_time'] = df_id.loc[ind_churn - 1, 'MONTH_PERIOD']
                for col in set(df_id.columns) - {'MONTH_PERIOD', 'CHURNED_IND', 'COMMERCIALLY_CHURNED', 'CUSTOMER_ID'}:
                    vals = df_id[ind_lt_1][col].values
                    _dics[id][col] = vals[0] # Most recent value
                    _dics[id][col + '_CHANGED'] = 1 if len(set(vals[1:horizon])) > 1 else 0
                    # _dics[id][col + '_PAST_STATE'] = lst[0] if len(lst[1:]) == 0 else max(vals[1:horizon], key=lst.count) # Most frequent value past horizon 
            else:
                _dics[id]['churn_event'] = 0
                _dics[id]['churn_time'] = pd.to_datetime('2013-1-1')
                for col in set(df_id.columns) - {'MONTH_PERIOD', 'CHURNED_IND', 'COMMERCIALLY_CHURNED', 'CUSTOMER_ID'}:
                    vals = df_id[col].values
                    _dics[id][col] = vals[0] # Most recent value
                    _dics[id][col + '_CHANGED'] = 1 if len(set(vals[1:horizon])) > 1 else 0
                    # _dics[id][col + '_PAST_STATE'] = lst[0] if len(lst[1:]) == 0 else max(vals[1:horizon], key=lst.count) # Most frequent value past horizon 
        except Exception as e:
            print(id, e)
        
    print(f'Number of clients that rejoined [churn -> active]: {len(ids_rejoin)}')
    print(f'Number of clients with rechurned [active -> churn -> active]: {len(ids_rechurn)}')
    return pd.DataFrame(_dics).T.rename_axis('id').reset_index()