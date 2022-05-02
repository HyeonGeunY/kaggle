import missingno
import pandas as pd


def checking_percent_of_null(df: pd.DataFrame, stage: str = "train"):
    print(f"Percent of NaN value for {stage}")
    for col in df.columns:
        print(
            f"column; {col:>12}\t Percent of NaN value: {100 * (df[col].isnull().sum() / df[col].shape[0]):.2f}"
        )


def get_meta_data(df: pd.DataFrame):
    data = []
    for f in df.columns:
        # Defining the role
        if f == 'Transported':
            role = 'target'
        else:
            role = 'input'
         
        # Defining the level
        if f in ["PassengerId", "HomePlanet", "Cabin", "Destination", "Name"]:
            level = 'nominal'
        elif f in ["CryoSleep", "VIP"]:
            level = 'binary'
        elif f in ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]:
            level = 'comtinous'
        elif f in ["Age"]:
            level = 'discrete'

        # Initialize keep to True for all variables except for id
        keep = True
        if f in ['PassengerId', "Name"]:
            keep = False

        # Defining the data type 
        dtype = df[f].dtype

        # Creating a Dict that contains all the metadata for the variable
        f_dict = {
            'varname': f,
            'role': role,
            'level': level,
            'keep': keep,
            'dtype': dtype
        }
        data.append(f_dict)
        
    
    meta = pd.DataFrame(data, columns=['varname', 'role', 'level', 'keep', 'dtype'])
    meta.set_index('varname', inplace=True)
    return meta
