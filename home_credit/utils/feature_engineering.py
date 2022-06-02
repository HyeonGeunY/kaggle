import pandas as pd
import numpy as np


def get_feature_counts(df, id_column, group_var, name_of_output_column):
    feature_count = (
        df.groupby(id_column, as_index=False)[group_var]
        .count()
        .rename(columns={group_var: name_of_output_column})
    )
    return feature_count


def merge_left(df_to: pd.DataFrame, df_from: pd.DataFrame, on: str, fix_null=False):
    """left merge 수행, NaN 값은 0으로 대체한다.

    Args:
        df_to (pd.DataFrame): 데이터가 추가될 데이터프레임
        df_from (pd.DataFrame): 추가할 데이터프레임
        on (str): key feature

    Returns:
        pd.DataFrame: 결과 DataFrame
    """
    df_to = df_to.merge(df_from, on=on, how="left")
    col = df_from.drop(on, axis=1).columns
    if fix_null:
        df_to[col] = df_to[col].fillna(0)

    return df_to


def get_agg(df, group_var, agg_info=["count", "mean", "max", "min", "sum"]):
    if not isinstance(agg_info, (str, list)):
        print("agg params must be list or str")
        return

    if isinstance(agg_info, str):
        agg_info = [agg_info]

    df_agg = df.groupby(group_var, as_index=False).agg(agg_info).reset_index()
    return df_agg


def stretch_columns(df, name_column: str, df_name: str = None):
    columns = [name_column]

    for var in df.columns.levels[0]:
        if var == name_column:
            continue

        if df_name:
            for stat in df.columns.levels[1][:-1]:  # 마지막은 빈 요소
                columns.append(f"{df_name}_{var}_{stat}")
        else:
            for stat in df.columns.levels[1][:-1]:  # 마지막은 빈 요소
                columns.append(f"{var}_{stat}")

    return columns


def get_r_coff(df, columns):
    # List of new correlations
    corr_list = []

    # Iterate through the columns
    for col in columns:
        # Calculate correlation with the target
        corr = df["TARGET"].corr(df[col])

        # Append the list as a tuple

        corr_list.append((col, corr))

    corr_list = sorted(corr_list, key=lambda x: abs(x[1]), reverse=True)

    return corr_list


def agg_numeric(df, group_var, df_name):
    """ 통계치 반환 과정 통합
    
    Parameters
    --------
        df (dataframe): 
            통계치를 계산한 데이터 프레임
        group_var (string): 
            group by를 수행할 기준 특징
        df_name (string): 
            바꿀 특징 이름 
        
    Return
    --------
        agg (dataframe): 
            a dataframe with the statistics aggregated for 
            all numeric columns. Each instance of the grouping variable will have 
            the statistics (mean, min, max, sum; currently supported) calculated. 
            The columns are also renamed to keep track of features created.
    
    """

    for col in df:
        if col != group_var and "SK_ID" in col:
            df = df.drop(columns=col)

    group_ids = df[group_var]
    numeric_df = df.select_dtypes("number")
    numeric_df[group_var] = group_ids

    agg = get_agg(numeric_df, group_var)
    columns = stretch_columns(agg, group_var, df_name)

    agg.columns = columns

    return agg


def count_categorical(df, group_var, df_name):
    """Computes counts and normalized counts for each observation
    of `group_var` of each unique category in every categorical variable
    
    Parameters
    --------
    df : dataframe 
        The dataframe to calculate the value counts for.
        
    group_var : string
        The variable by which to group the dataframe. For each unique
        value of this variable, the final dataframe will have one row
        
    df_name : string
        Variable added to the front of column names to keep track of columns

    
    Return
    --------
    categorical : dataframe
        A dataframe with counts and normalized counts of each unique category in every categorical variable
        with one row for every unique value of the `group_var`.
        
    """

    # Select the categorical columns
    categorical = pd.get_dummies(df.select_dtypes("object"))

    # Make sure to put the identifying id on the column
    categorical[group_var] = df[group_var]

    # Groupby the group var and calculate the sum and mean
    categorical = categorical.groupby(group_var).agg(["sum", "mean"])

    column_names = []

    # Iterate through the columns in level 0
    for var in categorical.columns.levels[0]:
        # Iterate through the stats in level 1
        for stat in ["count", "count_norm"]:
            # Make a new column name
            column_names.append(f"{df_name}_{var}_{stat}")

    categorical.columns = column_names

    return categorical
