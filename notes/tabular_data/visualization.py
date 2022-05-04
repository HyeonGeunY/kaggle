import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List

# bar
def plot_bar_sns(df: pd.dataFrame, x_feature: str, y_feature: str):
    plt.figure()
    fig, ax = plt.subplots(figsize=(20,10))
    
    sns.barplot(ax=ax, x=x_feature, y=y_feature, data=cat_perc, order=cat_perc[f])
    plt.ylabel(y_feature, fontsize=18)
    plt.xlabel(x_feature, fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.show()

def plot_bar_along_columns(df: pd.DataFrame, x: str = None, y: List[str] = None, title: str = "set_title"):
    
    f, ax = plt.subplots(1, 1, figsize=(20, 8))

    if not x:
        df.plot(y=y, kind="bar", ax=ax)
        plt.title(title, fontsize=20)
        
    else:
        df.plot(x=x, y=y, kind="bar", ax=ax)
        plt.title(title, fontsize=20)
    
    ## 눈금 글자 크기 지정
    plt.tick_params(axis='x', direction='in', length=3, pad=6, rotation=-90, labelsize=20, top=True)
    plt.tick_params(axis='y', direction='in', length=3, pad=6, labelsize=20, top=True)
    plt.grid()
    
    ## legend 크기 지정 (글꼴 그키 지정)
    params = {'legend.fontsize': 20,
          'legend.handlelength': 2}

    plt.rcParams.update(params)
    plt.savefig(title, format="png")
        
    
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
    plt.show()


def checking_binary_target_label_distribution(df: pd.DataFrame):
    f, ax = plt.subplots(1, 2, figsize=(18, 8))
    df['Transported'].value_counts().plot.pie(explode=[0, 0.1], autopct="%1.1f%%", ax=ax[0], shadow=True)
    ax[0].set_title('Pie plot - Transported')
    ax[0].set_ylabel('')
    sns.countplot('Transported', data=df, ax=ax[1])
    ax[1].set_title('Count plot - Transported')
    plt.show()
    
    return


def count_plot(df: pd.DataFrame, feature: str, hue: str):

    y_position = 1.02
    f, ax = plt.subplots(1, 2, figsize=(30, 8))
    df[feature].value_counts().plot.bar(color=['#CD7F32','#FFDF00'], ax=ax[0]) # color=['#CD7F32','#FFDF00','#D3D3D3']
    ax[0].set_title(f"Number of samples by {feature}", y=y_position)
    ax[0].set_ylabel("Count")
    sns.countplot(feature, hue=hue, data=df, ax=ax[1])
    ax[1].set_title(f"{feature} {hue} vs not", y=y_position)
    plt.subplots_adjust(wspace=0.3, hspace=0.5)

    # if need more col => add like below
    # sns.countplot("HomePlanet", hue="CryoSleep", data=df_train, ax=ax[2])
    # ax[2].set_title("CryoSleep by HomePlanet", y=y_position)
    # plt.subplots_adjust(wspace=0.3, hspace=0.5)
    plt.show()


def distplot_binary(df: pd.DataFrame, feature: str, hue: str):
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    sns.distplot(df[df[hue] == 0][feature], ax=ax)
    sns.distplot(df[df[hue] == 1][feature], ax=ax)
    plt.legend([f'{hue} == 0', f'{hue} == 1'])
    plt.title(f"{feature} vs {hue}")
    plt.show()


def factorplot(df: pd.DataFrame, feature: str, target: str, hue: str = None, col: str = None):

    if hue and col:
        sns.factorplot(feature, target, hue=hue, col=col, data=df, 
                size=6, aspect=1.5)
        plt.title(f"{feature} & {hue} vs {target}")
    elif hue and (not col):
        sns.factorplot(feature, target, hue=hue, data=df, 
                size=6, aspect=1.5)
        plt.title(f"{feature} & {hue} vs {target}")
    elif (not hue) and col:
        sns.factorplot(feature, target, col=col, data=df, 
                size=6, aspect=1.5)
        plt.title(f"{feature} & {hue} vs {target}")
    else:
        sns.factorplot(feature, target, data=df, 
                size=6, aspect=1.5)
        plt.title(f"{feature} vs {target}")