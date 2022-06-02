import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

COLOR = ["#CD7F32", "#FFDF00"]


class KDE():
    def __init__(self, x_labelsize=20, y_labelsize=20, title_size=30, legend_size=20):
        self.x_labelsize = x_labelsize
        self.y_labelsize = y_labelsize
        self.title_size = title_size
        self.legend_size = legend_size
    
    def __call__(self, df, var_name, target='TARGET'):
        
        # Calculate the correlation coefficient between the new variable and the target
        corr = df[target].corr(df[var_name])
        
        # Calculate medians for repaid vs not repaid
        avg_repaid = df[df[target] == 0][var_name].median()
        avg_not_repaid = df[df[target] == 1][var_name].median()
        
        plt.figure(figsize = (12, 6))
        
        # Plot the distribution for target == 0 and target == 1
        sns.kdeplot(df[df[target] == 0][var_name], label = 'TARGET == 0')
        sns.kdeplot(df[df[target] == 1][var_name], label = 'TARGET == 1')
        
        # label the plot
        plt.xlabel(var_name, fontsize=20) 
        plt.ylabel('Density', fontsize=20) 
        plt.title(f'{var_name} Distribution', size=30)
        plt.legend(fontsize=20)
        plt.tight_layout()
        
        # print out the correlation
        print(f'The correlation between {var_name} and the TARGET is %{corr:.4f}')
        # Print out average values
        print(f'Median value for loan that was not repaid = %{avg_not_repaid:.4f}')
        print(f'Median value for loan that was repaid = %{avg_repaid:.4f}')