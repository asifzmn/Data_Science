import json
from collections import Counter

import webcolors as webcolors
import xlrd
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, rgb2hex


def geojsonFile():
    with open('static/Geofiles/districts.json') as f:data = json.load(f)
    print(data.keys())
    print(data['features'][0].keys())
    for key in data['features'][0].values():
        print(type(key))
    print(data['features'][0]['geometry']['coordinates'])
    l = data['features'][0]['geometry']['coordinates']
    print(len(l[0][0]))
    # print(data['features'][0]['properties'])

def GetSulmData(update = False):
    if update:
        data = pd.read_csv('/home/az/Desktop/slumreport.csv', header=None)

        starts = data.index[data.loc[:, 1] == 'Zila name prior to coming to slum area'].tolist()[:-1]
        ends = data.index[data.loc[:, 1] == 'Total'].tolist()[:-1]

        districtData = []
        for s, e in zip(starts, ends):
            disdf, disdf.index.name = data.loc[s:e - 1, 1:2].set_index(1), None
            districtData.append(disdf.rename(columns=data.iloc[s - 2, 2:3]).drop(disdf.index[0]))
            # print(data.loc[s-2,1:2])

        df = pd.concat(districtData, axis=1, sort=False).fillna(0).assign(Lakshmipur=0).astype('int')
        df = df.rename(columns={"Chapai nababganj": "Nawabganj"},index={"Chapai nababganj": "Nawabganj"})
        df = df[sorted(df.columns)].sort_index()
        df.to_csv('BD_Slum.csv')
    df = pd.read_csv('BD_Slum.csv', header=0, index_col=0)
    return df

def colorMap(bins):
    newcolors = cm.get_cmap('OrRd')(np.linspace(0, 1, len(bins) + 4))[3:, :-1]
    print(repr([rgb2hex(col) for col in newcolors]))

if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    df = GetSulmData()
    bins = [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,20000,30000,40000,50000,100000]
    quantizedData = [np.digitize(x, bins) for x in df.values]
    dfCat = pd.DataFrame(data=quantizedData,index=df.index.values,columns=df.columns.values)
    # dfCat.T.to_csv('BD_Slum_Cat.csv')

    # fig = go.Figure(data=go.Heatmap(
    #     z=df.values,
    #     x=df.index.values,
    #     y=df.columns.values,
    #     colorscale='purpor'))
    #
    # fig.update_layout(title='Slum',xaxis_nticks=64,yaxis_nticks=64)
    # fig.show()

    colorMap(bins)
    # print(sorted(Counter(np.round(df.values.reshape(-1)/1000))))

    exit()
