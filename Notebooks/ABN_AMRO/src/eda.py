from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pyvis.network import Network
from lifelines.plotting import add_at_risk_counts

# def plot_churn_risk(df, col, estimator, T:str='churn_time', E:str='event',  horizon=12):
#     """ Plot univariate churn risk """
#     if col not in df.columns:
#         raise KeyError(f'{col} not in data frame')
#     fig, ax = plt.subplots(figsize=(6, 5))
#     timeline = np.linspace(0, horizon, 1000)
#     vals = set(df[col].values)
#     estimators = [estimator(label=col+'_'+str(val)) for val in vals]
#     inds = [df_red[col] == val for val in vals]
#     estimators = [estimator.fit(df_red[inds[i]][T], df_red[inds[i]][E], timeline=timeline) 
#                     for i, estimator in enumerate(estimators)]
#     # if type(estimator) in ['NelsonAalenFitter', 'WeibullFitter']:
#     [estimator.plot(ax=ax) for estimator in estimators]
#     add_at_risk_counts(*estimators, ax=ax, fontsize=10, rows_to_show=['At risk','Events'])
#     ax.set_ylim(0.7)
#     ax.set_xlabel('Months', fontsize=10)
#     ax.set_ylabel('Risk of churn', fontsize=10)

def plot_km_churn_risk(df, col, estimator, ax=None, T:str='churn_time', E:str='churn_event',  horizon=23, at_risk=False):
    """ Plot univariate churn risk """

    if col not in df.columns:
        raise KeyError(f'{col} not in data frame')
    if not ax:
        fig, ax = plt.subplots(figsize=(6, 5))
    timeline = np.linspace(0, horizon, 1000)
    vals = set(df[col].values)
    estimators = [estimator(label=col.split('_')[0]+'_'+str(val)) for val in vals]
    inds = [df[col] == val if str(val) != 'nan' else df[col].isna() for val in vals ]
    estimators = [estimator.fit(df[inds[i]][T], df[inds[i]][E], timeline=timeline) 
                    for i, estimator in enumerate(estimators)]
    # if type(estimator) in ['NelsonAalenFitter', 'WeibullFitter']:
    for estimator in estimators:
        estimator.plot(ci_show=False, ax=ax)
    if at_risk:
        add_at_risk_counts(*estimators, ax=ax, fontsize=10, rows_to_show=['At risk'])
    ax.set_ylim(0.5)
    ax.set_xlim([0, 23])
    ax.set_xlabel('Months', fontsize=10)
    ax.set_ylabel('Probability of Retention', fontsize=10)


def top_columns_with_missingness(df):
    serie = (
        df.isna()
        .sum()
        .sort_values(ascending=False) 
        * 100 
        / len(df)
    )
    return serie[serie > 0]


def plot_graph(corr):
    """ Plot an interactive graph using Networkx """
    _dict = defaultdict(set)
    for i, j in corr.index:
        _dict[i].add(j)
    G = nx.Graph()
    G.add_nodes_from(_dict.keys())
    for node, neighbours in _dict.items():
            G.add_edges_from(([(node, neighbour) for neighbour in neighbours]))
    net = Network(notebook=True, cdn_resources='remote')
    net.from_nx(G)
    net.show("/Users/Danial/Repos/ml/Notebooks/ABN_AMRO/graph/example.html")