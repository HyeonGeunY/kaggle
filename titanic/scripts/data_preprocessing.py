import pandas as pd
import numpy as np
import re


class titanic_features(): 
    def __init__(self, train_data):
        self.train = train_data
        
    def add_features_v1(self, dataset: pd.DataFrame = None):
        """훈련에 필요한 특징만을 남긴다.

        Args:
            data (pd.DataFrame, optional): _description_. Defaults to None.
            train 또는 test 데이터 셋
        """
        dataset = self.fill_null(dataset)
        # 이름 길이
        dataset['Name_length'] = dataset['Name'].apply(len)
        dataset['Has_Cabin'] = dataset["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
        dataset['IsAlone'] = 0
        dataset['Title'] = dataset['Name'].apply(self.get_title)
        dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
        
        dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    
        # Mapping titles
        title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
        dataset['Title'] = dataset['Title'].map(title_mapping)
        dataset['Title'] = dataset['Title'].fillna(0)
        
        # Mapping Embarked
        dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
        
        # Mapping Fare 
        # q_cut을 기준으로 데이터 수가 유사하게 그룹 나누기
        dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 						        = 0
        dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
        dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
        dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							        = 3
        dataset['Fare'] = dataset['Fare'].astype(int)
        
        # Mapping Age 
        # q_cut을 기준으로 데이터 수가 유사하게 그룹 나누기
        dataset.loc[ dataset['Age'] <= 16, 'Age'] 					       = 0
        dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
        dataset.loc[ dataset['Age'] > 64, 'Age']                           = 4
        
        dataset = self.drop_features_for_v1(dataset)
        return dataset
        
    def fill_null(self, dataset):
        train = self.train
        dataset['Embarked'] = dataset['Embarked'].fillna('S')
        dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
        
        age_avg = dataset['Age'].mean()
        age_std = dataset['Age'].std()
        age_null_count = dataset['Age'].isnull().sum()
        age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
        dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
        dataset['Age'] = dataset['Age'].astype(int)
        
        return dataset
        
    
    def get_title(self, name):
        # 이름의 형식이 , Mr. 로 되어 있기 때문에 앞쪽에 공백(' ')이 필요.
        title_search = re.search(' ([A-Za-z]+)\.', name)
        # If the title exists, extract and return it.
        if title_search:
            return title_search.group(1)
        return ""
    
    def drop_features_for_v1(self, dataset):
        drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']

        return dataset.drop(drop_elements, axis=1)
        
    
    
    

