import numpy as np
import pandas as pd

def add_v1_features(df):
    """모델 version 1에 사용할 특징을 추가한다.

    Args:
        df (pd.DataFrame): 사용할 데이터
        
    Returns:
        pd.DataFrame: 사용할 특징을 담은 데이터
    """
    v1_features = ["CryoSleep", "HomePlanet", "Age", "AmenitiesFare_1", "AmenitiesFare_2", "Destination", "Deck", "Side"]
    
    df["AmenitiesFare_1"] = df[["RoomService", "ShoppingMall"]].sum(axis=1).apply(lambda x: np.log(x) if x > 0 else 0)
    df["AmenitiesFare_2"] = df[["FoodCourt", "Spa", "VRDeck"]].sum(axis=1).apply(lambda x: np.log(x) if x > 0 else 0)
    
    df["Deck"] = df["Cabin"].str.split("/").str[0]
    df["Side"] = df["Cabin"].str.split("/").str[2]
    
    return df[v1_features]


def fill_cryosleep_null(x):
    
    if pd.isnull(x["CryoSleep"]):
        # 어메니티에 사용한 금액이 없는 경우
        if x[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum() == 0:
            return True
        # 목적지가 "TRAPPIST-1e" 인 경우
        elif x["Destination"] == "TRAPPIST-1e":
            return False
        # 혼자 여행 or 출신지가 Earth인 경우
        elif x["GroupSize"] == 1 or x["HomePlanet"] == "Earth":
            return False
        else:
            return True
    else:
        return x["CryoSleep"]

def fill_homeplanet_null(x):
    if pd.isnull(x["HomePlanet"]):
        # Deck == "F" or "D"
        if x["Deck"] in ['F', 'D']:
            return "Mars"
        # Deck in 
        elif x["Deck"] in ["A", "B", "C", "D", "E", "T"] or x['Destination'] == '55 Cancri e':
            return "Europa"
        else:
            return 'Earth'
    else:
        return x["HomePlanet"]