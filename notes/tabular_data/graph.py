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
    
    
def corr_heatmap(df: pd.DataFrame, features: List[str]):
    correlations = df[features].corr()

    # Create color map ranging between two colors
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(correlations, cmap=cmap, vmax=1.0, center=0, fmt='.2f',
                square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .75})
    plt.show();