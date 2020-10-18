import h5py
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import os
from os import path

mainpath = '/media/az/Study/NTL Docs/Data/'

def DownloadFiles():
    df = pd.read_csv('/home/az/Desktop/LAADS_query.2020-08-09T14_42.csv')
    df.index = pd.date_range(start='2013-01-01', end='2013-12-31', freq='D')
    selection = (df.iloc[(df.index.day == 11) | (df.index.day == 21)])

    for i, row in selection.iterrows():
        filename = mainpath + i.strftime('%Y-%m-%d') + '.h5'
        if path.exists(filename): continue
        response = requests.get('https://ladsweb.modaps.eosdis.nasa.gov' + row['fileUrls for custom selected'],
                                allow_redirects=True)
        open(filename,'wb').write(response.content)
        print(filename)


def getavgFactors(f):
    dict = f['HDFEOS']['GRIDS']['VNP_Grid_DNB']['Data Fields']
    print(dict.keys())
    # return [np.mean(np.array(dict[key])) for key in ['Radiance_M10','QF_VIIRS_M10']]
    return [np.mean(np.array(dict[key])) for key in dict.keys()]


def visualize(f):
    dict = f['HDFEOS']['GRIDS']['VNP_Grid_DNB']['Data Fields'].items()
    # dict = itertools.islice(dict, 0, 4)
    # dict = itertools.islice(dict,13, 26)
    # dict = itertools.islice(dict,13, 21)

    # for k, v in dict:
    #     # print(k)
    #     histo = (np.array(v))
    #     print(k,np.mean(histo))
    #     # print(histo.shape)
    #     # print(np.min(np.array(v)), np.max(np.array(v)))
    #
    #     # fig = go.Figure(data=go.Heatmap(
    #     #     z=histo,
    #     #     colorscale='blues',
    #     #     # reversescale=False
    #     # ))
    #     # fig.update_layout(title=k, xaxis_title="Users", yaxis_title="Time")
    #     # fig.show()


if __name__ == '__main__':

    DownloadFiles()
    exit()

    files = [mainpath + "/" + f for f in os.listdir(mainpath)]

    lst = []
    for month in range(1, 13):
        fname = '/home/az/Desktop/NTL Docs/DL/2013-' + str(month).zfill(2) + '-01.h5'
        f = h5py.File(fname, 'r')
        lst.append(getavgFactors(f))

    lst = np.array(lst).T
    print(lst.shape)

    fig = go.Figure()
    for ll in lst:
        fig.add_trace(go.Scatter(x=np.arange(1, 13), y=ll, mode='lines+markers'))
    fig.update_layout(font_size=18, title='Timeseries of average usage', xaxis_title="Time", yaxis_title="Unit")
    fig.show()

    # print(type(f))
    # print(type(f['HDFEOS']))
    # print(type(f['HDFEOS INFORMATION']))
    #
    # print(f.keys())
    # print(f['HDFEOS'].keys())
    # print(f['HDFEOS INFORMATION'].keys())
    #
    # print(f['HDFEOS']['ADDITIONAL']['FILE_ATTRIBUTES'])
    # print(f['HDFEOS']['GRIDS']['VNP_Grid_DNB']['Data Fields'].keys())

    # print(np.array(f['HDFEOS']['GRIDS']['VNP_Grid_DNB']['Data Fields']['UTC_Time']))

    exit()
