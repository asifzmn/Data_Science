import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('/home/az/Desktop/Hazard Map BD/hazard',sep=' ',skiprows=0,index_col='District_name')
    print(df.to_string())
    print(df.columns)
    df.to_csv('hazard.csv')

    df1 = pd.read_csv('/home/az/Desktop/Hazard Map BD/hazard_score',sep=' ',skiprows=0,index_col='District_name').sort_values('District_name')
    print(df1.to_string())
    print(df1.columns)
    print(df1.describe())
    df1.to_csv('hazard_score.csv')
