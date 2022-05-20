import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno


class TableDataViz():
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.colors=["#CD7F32", "#FFDF00"]
        self.meta = None
        
    # 특징 고윳값별 개수 시각화
    def countplot_sns(self, feature: str):
        plt.figure()
        fig, ax = plt.subplots(figsize=(20, 10))
        sns.countplot(x=feature, data=self.df, ax=ax)
        ax.set_title(f"{feature}_distribution", fontsize=30)
        self.set_label(ax, x_label="target", y_label="Count", fontsize=20)
        self.write_percent(ax=ax, total_size=len(self.df[feature]))
        plt.show()
        
        
    # 이진 데이터 타겟 비율 시각화
    def plot_target_ratio_by_features(self, features, num_rows, num_cols, size=None):
        
        if not size:
            size = (num_cols*9, num_rows*7)
        
        fig, ax = plt.subplots(num_rows, num_cols, figsize=size)
        
        
        for i in range(num_rows):
            for j in range(num_cols):
                sns.barplot(x=features[(i * num_cols) + j], y='target', data=self.df, ax=ax[i][j])
                
        plt.tight_layout()
        
    
    def plot_target_ratio_by_features_continous(self, features, num_rows, num_cols, size=None, n_cut=5):
        
        if not size:
            size = (num_cols*9, num_rows*7)
        
        fig, ax = plt.subplots(num_rows, num_cols, figsize=size)
    
        for i in range(num_rows):
            for j in range(num_cols):
                sns.barplot(x=pd.cut(features[(i * num_cols) + j], n_cut), y='target', data=self.df, ax=ax[i][j])
                
        plt.tight_layout()
    
    
    # msno를 이용한 결측치 시각화
    def plot_msno_bar(self, start_idx: int = None, end_idx: int = None, figsize=(13, 6)):
        if start_idx and end_idx:
            msno.bar(df=self.df.iloc[:, start_idx:end_idx])
        elif start_idx and not end_idx:
            msno.bar(df=self.df.iloc[:, start_idx:])
        elif not start_idx and end_idx:
            msno.bar(df=self.df.iloc[:, :end_idx])
        else:
            msno.bar(df=self.df.iloc[:, :])
            
            
    def plot_msno_matrix(self, start_idx: int = None, end_idx: int = None, figsize=(13, 6)):
        if start_idx and end_idx:
            msno.matrix(df=self.df.iloc[:, start_idx:end_idx])
        elif start_idx and not end_idx:
            msno.matrix(df=self.df.iloc[:, start_idx:])
        elif not start_idx and end_idx:
            msno.matrix(df=self.df.iloc[:, :end_idx])
        else:
            msno.matrix(df=self.df.iloc[:, :])
    
        
    def set_label(self, ax, x_label: str = '', y_label: str = '', fontsize=20):
        if x_label:
            ax.set_xlabel(x_label, fontsize=fontsize)
        if y_label:
            ax.set_ylabel(y_label, fontsize=fontsize)
        
    
    # 막대 그래프 위에 비율 표시
    def write_percent(self, ax, total_size, fontsize=20):
        for patch in ax.patches:
            height = patch.get_height() # height (num of data)
            width = patch.get_width()
            left_coord = patch.get_x() # x coord of left side of geo
            percent = height/total_size*100
        
            # write percent info at (x, y)
            ax.text(left_coord + width/2.0, height + total_size*0.001,f"{percent:1.1f}", ha='center', fontsize=20)
    
    
class MetaTable():
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.meta=None
        self._first_meta_table()
        
    #초기 meta 데이터 생성
    def _first_meta_table(self):
        if self.meta:
            print("Already exists")
            return self.meta
        
        print(f"dataset shape: {self.df.shape}")
        summary = pd.DataFrame(self.df.dtypes, columns=['data_type'])
        summary['num_of_NaN'] = (self.df == -1).sum().values
        summary['num_of_unique_value'] = self.df.nunique().values
        summary['data_category'] = None
        summary['keep'] = True
        summary['role'] = 'input'
    
        for col in self.df.columns:
            if 'bin' in col or col == 'target':
                summary.loc[col, 'data_category'] = 'binary'
                summary.loc[col, 'role'] = 'target'
            elif 'cat' in col:
                summary.loc[col, 'data_category'] = 'nominal'
            elif self.df[col].dtype == float:
                summary.loc[col, 'data_category'] = 'continuous'
            elif self.df[col].dtype == int:
                summary.loc[col, 'data_category'] = 'ordinal'
        
        self.meta = summary
        return self.meta