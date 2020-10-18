from collections import Counter

import pandas as pd

if __name__ == '__main__':
    filename, sheetname = '/home/az/Desktop/News/Digital Government&Pandemic 28 April 2020.xlsx', 'WEB CRAWLER'
    wordfile = pd.read_excel(filename, sheet_name=sheetname, skiprows=0)
    # print(wordfile['Unnamed: 3'].values)

    newsData = pd.read_csv('/home/az/Desktop/News/new_york_times_news_data_.csv', engine='python', header=None)
    # print(newsData.loc[:,3])
    # print(newsData.loc[:,4])
    # print(newsData.loc[0,5].encode('ascii',errors='ignore').decode('UTF-8'))

    l = []
    for word in wordfile['Unnamed: 3'].values:
        for i in range(100):
            idx = newsData.loc[i, 5].encode('ascii', errors='ignore').decode('UTF-8').find(word)
            if not idx == -1: l.append(word)

    print(Counter(l))
    exit()
