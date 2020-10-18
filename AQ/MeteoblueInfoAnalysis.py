import os
from datetime import datetime, timedelta
import time
from os import listdir
from os.path import isfile, join

from pandas_profiling import ProfileReport

from AQ.DataPreparation import LoadMetadata
from timeit import default_timer as timer
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import xarray as xr

attribSel = ['Relative Humidity  [2 m above gnd]', 'Temperature  [2 m above gnd]', 'Wind Gust  [sfc]',
             'Total Cloud Cover  [sfc]', 'Sunshine Duration  [sfc]']

factors = ['Temperature [2 m]', 'Relative Humidity [2 m]', 'Mean Sea Level Pressure', 'Precipitation',
           'Cloud Cover High', 'Cloud Cover Medium', 'Cloud Cover Low', 'Sunshine Duration', 'Shortwave Radiation',
           'Direct Shortwave Radiation', 'Diffuse Shortwave Radiation', 'Wind Gust', 'Wind Speed [10 m]',
           'Wind Direction [10 m]', 'Wind Speed [80 m]', 'Wind Direction [80 m]', 'Wind Speed [900 mb]',
           'Wind Direction [900 mb]', 'Wind Speed [850 mb]', 'Wind Direction [850 mb]', 'Wind Speed [700 mb]',
           'Wind Direction [700 mb]', 'Wind Speed [500 mb]', 'Wind Direction [500 mb]', 'Temperature [1000 mb]',
           'Temperature [850 mb]', 'Temperature [700 mb]', 'Surface Temperature', 'Soil Temperature [0-10 cm down]',
           'Soil Moisture [0-10 cm down]']


def WindDirFactors():
    windDirNames = np.array(
        ['Wind Direction [10 m]', 'Wind Direction [80 m]', 'Wind Direction [500 mb]', 'Wind Direction [700 mb]',
         'Wind Direction [850 mb]', 'Wind Direction [900 mb]'])
    # colorPal = np.array(['#ffffb1', '#ffd500', '#ffb113', '#ADD8E6', '#87CEEB', '#1E90FF'])
    colorPal = np.array(['#C0C0C0', '#C0C0C0', '#C0C0C0', '#C0C0C0', '#C0C0C0', '#C0C0C0'])
    return windDirNames, colorPal


def getSides(x, seg=16):
    directon = np.floor((x - (360 / seg) / 2) / (360 / seg)).astype('int')
    directon[directon == -1] = seg - 1
    return np.roll(np.hstack((np.bincount(directon), np.zeros(seg - max(directon) - 1))), 1)


# def WindGraph(x): PlotlyRosePlot([[getSides(x), 'Wind', '#3f65b1']])

def PrepareMeteoData2(file):
    def dates(file):
        dates = file.split('/')[-2]
        time = pd.to_datetime(dates.split(' to '))
        return pd.date_range(start=time.min(), end=time.max() + timedelta(days=1), freq='H')[:-1]

    meteoInfo = pd.read_csv(file, sep=',', skiprows=9).replace(-999, 0)
    meteoInfo.drop(['timestamp'], axis=1, inplace=True)
    meteoInfo = meteoInfo.apply(pd.to_numeric)
    meteoInfo.columns, meteoInfo.index = factors, dates(file)
    meteoInfo.columns.name, meteoInfo.index.name = 'Factors', 'Time'
    meteoInfo.iloc[:, 1] = meteoInfo.iloc[:, 1] / 100
    return meteoInfo.stack()


def PrepareMeteoDatatime(file): return pd.to_datetime(pd.read_csv(file, sep=',', skiprows=9)['timestamp'])


def PrepareMeteoData1(file):
    meteoInfo = pd.read_csv(file, sep=',', skiprows=9).replace(-999, 0)
    meteoInfo.drop(['timestamp'], axis=1, inplace=True)
    meteoInfo = meteoInfo.apply(pd.to_numeric)
    meteoInfo.iloc[:, 1] = meteoInfo.iloc[:, 1] / 100
    return meteoInfo.values


def MeteoTimeSeries(meteoData, factorname, unit, colors):
    fig = go.Figure()
    windDirNames = WindDirFactors()[0]
    for i, factor in enumerate(np.setdiff1d(meteoData['factor'].values, windDirNames)): fig.add_trace(
        go.Scatter(x=pd.Series(meteoData['time'].values), y=meteoData.loc[:, factor], name=factor,
                   marker_color=colors[i]))  # line_color='deepskyblue','dimgray'
    fig.update_traces(mode='lines+markers', marker=dict(line_width=0, symbol='circle', size=3))
    fig.update_layout(title=factorname, font_size=16, legend_font_size=16, template="ggplot2",
                      xaxis_title="Time", yaxis_title=unit)
    # fig.update_layout(title_text='Time Series with Rangeslider',xaxis_rangeslider_visible=True)
    fig.show()


def oneFolder(locationMain, folders):
    time = pd.to_datetime(folders.split(' to '))
    time = pd.date_range(start=time.min(), end=time.max() + timedelta(days=1), freq='H')[:-1]

    meteoData = np.array(
        [PrepareMeteoData1(locationMain + '/' + folders + '/' + district + '.csv') for district in metaFrame.index])
    return xr.DataArray(data=meteoData,
                        coords={"district": metaFrame.index.values, "time": time, "factor": factors},
                        dims=["district", "time", "factor"], name='meteo')


def GetAllMeteoData():
    locationMain = '/media/az/Study/Air Analysis/Dataset/Meteoblue Scrapped Data'
    return xr.merge([oneFolder(locationMain, folders) for folders in listdir(locationMain)[:]])
    # for folders in listdir(locationMain)[:3]:
    #     print(oneFolder(locationMain, folders))


def PlotlyRosePlot(info, colorPal, alldistricts):
    fig = go.Figure()
    # for [r,name] in info:fig.add_trace(go.Barpolar(r=r,name=name,marker_color=colorPal))
    for infod in info:
        for [r, name], colorp in zip(infod, colorPal):
            fig.add_trace(go.Barpolar(r=r, name=name, marker_color=colorp))

    districtCount, windStaes = info.shape[0], info.shape[1]
    states = np.full((districtCount, districtCount * windStaes), False, dtype=bool)
    for i in range(districtCount): states[i][windStaes * i:windStaes * (i + 1)] = True

    fig.update_layout(
        updatemenus=[
            dict(
                buttons=list([
                    dict(label=district, method="update",
                         args=[{"visible": state}, {"title": "Wind direction in " + district}])
                    for district, state in zip(alldistricts, states)]),
                active=0
            )
        ])

    fig.update_traces(
        text=['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'])
    # fig.update_traces(text=['North', 'N-E', 'East', 'S-E', 'South', 'S-W', 'West', 'N-W'])
    fig.update_layout(
        title='Wind', font_size=16, legend_font_size=16, polar_radialaxis_ticksuffix='', polar_angularaxis_rotation=90,
        template="plotly_dark", polar_angularaxis_direction="clockwise"
    )
    fig.show()


def WindGraphTeam(meteoData):
    windDirNames, colorPal = WindDirFactors()
    alldis = meteoData['districts'].values
    directions = np.array(
        # [[[getSides(meteoData.loc[dis, '2020-03-29':'2020-03-31', factor].values), factor] for factor in windDirNames]
        [[[getSides(meteoData.loc[dis, :, factor].values), factor] for factor in windDirNames]
         for dis in alldis])
    PlotlyRosePlot(directions, colorPal, alldis)


def MeteoBoxPlot(meteoData):
    factor = [['Surface Temperature', 'indianred'], ['Relative Humidity [2 m]', 'skyblue']][1]
    fig = go.Figure()
    for dis in metaFrame.index.values: fig.add_trace(
        go.Box(y=meteoData.loc[dis, :, factor[0]], name=dis, marker_color=factor[1]))
    fig.show()


selFactors = ['Temperature [2 m]', 'Surface Temperature', 'Soil Temperature [0-10 cm down]', 'Temperature [700 mb]',
              'Temperature [850 mb]', 'Temperature [1000 mb]']
factorUnit = 'Temperature', 'Celcius', ['#260637', '#843B58', '#B73239', '#FFA500', '#F9C53D', '#EADAA2', ]

# selFactors = ['Relative Humidity [2 m]','Soil Moisture [0-10 cm down]']
# factorUnit = 'Humidity','Fraction',['mediumblue','lightblue']

rad = ['Shortwave Radiation', 'Direct Shortwave Radiation', 'Diffuse Shortwave Radiation']
speed = ['Wind Speed [10 m]', 'Wind Speed [80 m]', 'Wind Speed [500 mb]', 'Wind Speed [700 mb]', 'Wind Speed [850 mb]',
         'Wind Speed [900 mb]']


def DateContinuity(df):
    all = pd.Series(data=pd.date_range(start=df[0].min(), end=df[0].max(), freq='M'))
    mask = all.isin(df[0].values)
    print(all[~mask])


def oneFolderAnother(locationMain, folders):
    time = pd.to_datetime(folders.split(' to '))
    time = pd.date_range(start=time.min(), end=time.max() + timedelta(days=1), freq='H')[:-1]

    meteoData = np.array(
        [PrepareMeteoData1(locationMain + '/' + folders + '/' + district + '.csv') for district in metaFrame.index])

    return xr.DataArray(data=meteoData,
                        coords={"district": metaFrame.index.values, "time": time, "factor": factors},
                        dims=["district", "time", "factor"], name='meteo')


def getFactorData(meteoData, factor):
    return meteoData.sel(factor=factor).to_dataframe().drop('factor',
                                                                                              axis=1).unstack().T.droplevel(
    level=0)


if __name__ == '__main__':
    metaFrame = LoadMetadata()

    # meteoData = GetAllMeteoData()
    # meteoData.to_netcdf('meteoData.nc')

    meteoData = xr.open_dataset('meteoData.nc')['meteo']

    # print(meteoData)
    # print(meteoData.shape)
    # print(meteoData.dims)
    # print(meteoData.coords)
    # for dim in meteoData.dims: print(meteoData.coords[dim])
    # print(meteoData.loc[:,'2020-07-02':'2020-07-05',['Temperature [2 m]','Precipitation']])
    # print(meteoData.sel(factor='Temperature [2 m]'))

    factor = 'Temperature [2 m]'
    df = getFactorData(meteoData, factor)
    print(df)
    print(df.index)
    print(df.columns)

    # print(meteoData.to_dataframe())

    # print(meteoData.loc['Azimpur'].to_dataframe(name='value'))

    # meteoData = xr.open_dataset('meteoData.nc').to_dataframe()
    # df = meteoData.loc['Azimpur'].unstack().T
    # df.index = df.index.droplevel(0)

    # prof = ProfileReport(df, minimal=False,title='Meteo Data')
    # prof.to_file(output_file='Meteo Data.html')

    # print(np.array_equal(combined.values,meteoData.values))
    # print(np.array_equal(merged.values,meteoData.values))

    # MeteoTimeSeries(meteoData.loc["Dhaka",:,selFactors],factorUnit[0],factorUnit[1],factorUnit[2])
    # MeteoBoxPlot(meteoData)
    # WindGraphTeam(meteoData)

    # for meteoInfo in meteoData[:1]:
    #     # dayGroup = meteoInfo.iloc[:72].groupby(meteoInfo.iloc[:72].index.day)
    #     dayGroup = meteoInfo.groupby(meteoInfo.index.day)
    #     for dayData in list(dayGroup)[::]:WindGraphTeam(dayData[1],metaFrame['Zone'].values)

    exit()
