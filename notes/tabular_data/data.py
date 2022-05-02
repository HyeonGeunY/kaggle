import missingno as msno
import pandas as pd

# null
def visualize_null_features(df: pd.DataFrame, start_idx: int = 0, end_idx: int = 0):

    if end_idx:
        msno.matrix(df=df.iloc[:, start_idx:end_idx], figsize=(20, 14), color=(0.42, 0.1, 0.05))
    else:
        msno.matrix(df=df.iloc[:, start_idx], figsize=(20, 14), color=(0.42, 0.1, 0.05))


def visualize_null_features_bar(df: pd.DataFrame, start_idx: int = 0, end_idx: int = 0):

    if end_idx:
        msno.bar(df=df.iloc[:, start_idx:end_idx], figsize=(20, 14), color=(0.42, 0.1, 0.05))
    else:
        msno.bar(df=df.iloc[:, start_idx], figsize=(20, 14), color=(0.42, 0.1, 0.05))


def checking_percent_of_null(df: pd.DataFrame, stage: str = "train"):
    print(f"Percent of NaN value for {stage}")
    for col in df.columns:
        print(
            f"column; {col:>12}\t Percent of NaN value: {100 * (df[col].isnull().sum() / df[col].shape[0]):.2f}"
        )


# meta data
def get_meta_data(df: pd.DataFrame):
    data = []
    for f in df.columns:
        # Defining the role
        if f == "Transported":
            role = "target"
        else:
            role = "input"

        # Defining the level
        if f in ["PassengerId", "HomePlanet", "Cabin", "Destination", "Name"]:
            level = "nominal"
        elif f in ["CryoSleep", "VIP"]:
            level = "binary"
        elif f in ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]:
            level = "continous"
        elif f in ["Age"]:
            level = "discrete"

        # Initialize keep to True for all variables except for id
        keep = True
        if f in ["PassengerId", "Name"]:
            keep = False

        # Defining the data type
        dtype = df[f].dtype

        # Creating a Dict that contains all the metadata for the variable
        f_dict = {"varname": f, "role": role, "level": level, "keep": keep, "dtype": dtype}
        data.append(f_dict)

    meta = pd.DataFrame(data, columns=["varname", "role", "level", "keep", "dtype"])
    meta.set_index("varname", inplace=True)
    return meta


def add_row_to_metadata(meta: pd.DataFrame, df, level: List[str], role="input", keep=True):
    """
    add row to meta data

    Parameters:
    meta(pd.DataFrame): meta data
    df(pd.Series or pd.DataFrame): df added to meta

    Returns:
    meta
    """
    level_group = ["ordinal", "nominal", "continuous", "binary"]

    if type(df) == pd.core.series.Series:
        if level not in level_group:
            raise f"level should be in [ordinal, nominal, continuous, binary]"

        name = df.name
        if name in meta.index:
            print(f"{name} already exists")
            return meta

        dtype = df.dtype
        meta = pd.concat(
            [
                meta,
                pd.DataFrame(
                    [[role, level, keep, dtype]],
                    columns=["role", "level", "keep", "dtype"],
                    index=[name],
                ),
            ],
            axis=0,
        )

    if type(df) == pd.core.frame.DataFrame:
        columns = df.columns

        for c, l in zip(columns, level):
            if l not in level_group:
                raise f"level should be in [ordinal, nominal, continuous, binary]"

            name = df[c].name
            if name in meta.index:
                print(f"{name} already exists")
                continue

            dtype = df[c].dtype
            meta = pd.concat(
                [
                    meta,
                    pd.DataFrame(
                        [[role, l, keep, dtype]],
                        columns=["role", "level", "keep", "dtype"],
                        index=[name],
                    ),
                ],
                axis=0,
            )

    return meta
