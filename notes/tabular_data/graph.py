import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List

def plot_bar_sns(df: pd.dataFrame, x_feature: str, y_feature: str):
    plt.figure()
    fig, ax = plt.subplots(figsize=(20,10))
    
    sns.barplot(ax=ax, x=x_feature, y=y_feature, data=cat_perc, order=cat_perc[f])
    plt.ylabel(y_feature, fontsize=18)
    plt.xlabel(x_feature, fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.show();
    
    
def corr_heatmap(df: pd.DataFrame, features: List[str] = None):
    
    if features:
        correlations = df[features].corr()
    else:
        correlations = df.corr()

    # Create color map ranging between two colors
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(correlations, cmap=cmap, vmax=1.0, center=0, fmt='.2f',
                square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .75})
    plt.title('Pearson correlation of continuous features', y=1.05, size=15)
    plt.show();
    
    
    
#barplot
## plot.ly
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

data = [go.Bar(
            x = train["target"].value_counts().index.values,
            y = train["target"].value_counts().values,
            text='Distribution of target variable'
    )]

layout = go.Layout(
    title='Target variable distribution'
)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='basic-bar')