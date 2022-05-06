import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List

## plot.ly
import plotly.offline as py

py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

# barplot

def plot_bar_sns(df: pd.dataFrame, x_feature: str, y_feature: str):
    plt.figure()
    fig, ax = plt.subplots(figsize=(20, 10))

    sns.barplot(ax=ax, x=x_feature, y=y_feature, data=cat_perc, order=cat_perc[f])
    plt.ylabel(y_feature, fontsize=18)
    plt.xlabel(x_feature, fontsize=18)
    plt.tick_params(axis="both", which="major", labelsize=18)
    plt.show()


def plot_bar_along_columns(
    df: pd.DataFrame, x: str = None, y: List[str] = None, title: str = "set_title"
):

    f, ax = plt.subplots(1, 1, figsize=(20, 8))

    if not x:
        df.plot(y=y, kind="bar", ax=ax)
        plt.title(title, fontsize=20)

    else:
        df.plot(x=x, y=y, kind="bar", ax=ax)
        plt.title(title, fontsize=20)

    ## 눈금 글자 크기 지정
    plt.tick_params(axis="x", direction="in", length=3, pad=6, rotation=-90, labelsize=20, top=True)
    plt.tick_params(axis="y", direction="in", length=3, pad=6, labelsize=20, top=True)
    plt.grid()

    ## legend 크기 지정 (글꼴 그키 지정)
    params = {"legend.fontsize": 20, "legend.handlelength": 2}

    plt.rcParams.update(params)
    plt.savefig(title, format="png")


def barplot_binary_plotly(df: pd.DataFrame):
    """
    visualize distribution of 0 and 1 values for each binary features
    
    df(pd.DataFrame): dataframe consist of binary features
    """

    zero_list = []
    one_list = []
    for col in df.columns:
        zero_list.append((train[col] == 0).sum())
        one_list.append((train[col] == 1).sum())

    trace1 = go.Bar(x=bin_col, y=zero_list, name="Zero count")
    trace2 = go.Bar(x=bin_col, y=one_list, name="One count")

    data = [trace1, trace2]
    layout = go.Layout(barmode="stack", title="Count of 1 and 0 in binary variables", title_x=0.5)

    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename="stacked-bar")


def barplot_plotly(df: pd.DataFrame, feature: str = "target"):
    data = [
        go.Bar(
            x=df[feature].value_counts().index.values,
            y=df[feature].value_counts().values,
            text=f"Distribution of {feature} variable",
        )
    ]

    layout = go.Layout(title=f"{feature} variable distribution")

    fig = go.Figure(data=data, layout=layout)

    py.iplot(fig, filename=f"basic-bar: {feature}")


# barh

def barh_plotly(x: List[str], y: List[float], title:str = "title"):
    
    trace = go.Bar(
        x=x ,
        y=y,
        marker=dict(
            color=x,
            colorscale = 'Viridis',
            reversescale = True
        ),
        name='Random Forest Feature importance',
        orientation='h',
    )

    layout = dict(
        title=title,
         width = 900, height = 2000,
        yaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=True,
    #         domain=[0, 0.85],
        ))

    fig = go.Figure(data=[trace])
    fig['layout'].update(layout)
    py.iplot(fig, filename='plots')




def corr_heatmap(df: pd.DataFrame, features: List[str] = None):

    if features:
        correlations = df[features].corr()
    else:
        correlations = df.corr()

    # Create color map ranging between two colors
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(
        correlations,
        cmap=cmap,
        vmax=1.0,
        center=0,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        annot=True,
        cbar_kws={"shrink": 0.75},
    )
    plt.title("Pearson correlation of continuous features", y=1.05, size=15)
    plt.show()


def checking_binary_target_label_distribution(df: pd.DataFrame):
    f, ax = plt.subplots(1, 2, figsize=(18, 8))
    df["Transported"].value_counts().plot.pie(
        explode=[0, 0.1], autopct="%1.1f%%", ax=ax[0], shadow=True
    )
    ax[0].set_title("Pie plot - Transported")
    ax[0].set_ylabel("")
    sns.countplot("Transported", data=df, ax=ax[1])
    ax[1].set_title("Count plot - Transported")
    plt.show()

    return


def count_plot(df: pd.DataFrame, feature: str, hue: str):

    y_position = 1.02
    f, ax = plt.subplots(1, 2, figsize=(30, 8))
    df[feature].value_counts().plot.bar(
        color=["#CD7F32", "#FFDF00"], ax=ax[0]
    )  # color=['#CD7F32','#FFDF00','#D3D3D3']
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
    plt.legend([f"{hue} == 0", f"{hue} == 1"])
    plt.title(f"{feature} vs {hue}")
    plt.show()


def factorplot(df: pd.DataFrame, feature: str, target: str, hue: str = None, col: str = None):

    if hue and col:
        sns.factorplot(feature, target, hue=hue, col=col, data=df, size=6, aspect=1.5)
        plt.title(f"{feature} & {hue} vs {target}")
    elif hue and (not col):
        sns.factorplot(feature, target, hue=hue, data=df, size=6, aspect=1.5)
        plt.title(f"{feature} & {hue} vs {target}")
    elif (not hue) and col:
        sns.factorplot(feature, target, col=col, data=df, size=6, aspect=1.5)
        plt.title(f"{feature} & {hue} vs {target}")
    else:
        sns.factorplot(feature, target, data=df, size=6, aspect=1.5)
        plt.title(f"{feature} vs {target}")


# scatter plot

def scatter_plotly(x: List[str], y: List[float], title: str = "title", ylabel: str = "ylabel"):
    trace = go.Scatter(
        y = y,
        x= x,
        mode='markers',
        marker=dict(
            sizemode = 'diameter',
            sizeref = 1,
            size = 13,
            color = y,
            colorscale='Portland',
            showscale=True
    ),
    text = x
        )
    
    data = [trace]
    
    layout= go.Layout(
        autosize=True,
        title=title,
        title_x=0.5,
        hovermode='closest',
        xaxis= dict(
            ticklen= 5,
            showgrid=False,
            zeroline=False,
            showline=False
        ),
        
        yaxis=dict(
        title= ylabel,
        showgrid=False,
        zeroline=False,
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
    )
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename='scatter')