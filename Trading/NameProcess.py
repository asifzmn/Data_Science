import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv("Company Info - Sheet1.csv", sep=',', skiprows=0)
    # for i  in (df.columns.values):print(i)
    # print(df)
    print(df['Company Type'].values)
    # print(df['Company Type'].value_counts())

    # ax = sns.countplot(x="Company Type", data=df)
    # plt.show()
