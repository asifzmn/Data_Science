import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

month_names = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
               'November', 'December']
monthColors = ["#8fcadd"] * 2 + ["#46d246"] * 3 + ["#ff0000"] * 3 + ["#ffa500"] * 3 + ["#8fcadd"]
monthcolorDict = dict(zip(month_names, monthColors))

mergePath = '/home/asif/Datasets/Electricity/Combined/'
all_zones = os.listdir(mergePath)
all_zones_path = [mergePath + zone for zone in all_zones]


def UsageTS(all_zones_path):
    # for zone, zone_path in zip(all_zones, all_zones_path):
    #     # print(zone)
    #     df = read_time_series_data(zone_path)
    #     print(zone, df.shape[1])
    #     # df["2013":"2016"].to_csv(mergePath+zone)
    #     # PlotUsageTimeseries(df, zone)

    for zone, zone_path in zip(all_zones, all_zones_path):
        # print(zone)
        df = read_time_series_data(zone_path)
        print(zone, df.shape[1])


def getAlldf(all_zones_path):
    dfs = [read_time_series_data(path) for path in all_zones_path]
    return pd.concat(dfs, axis=1)


def read_time_series_data(path): return pd.read_csv(path, index_col='Time', low_memory=False, parse_dates=[0])


def create_save_time_series(df, areaName):
    def dateSeries(x, date_rng): return x.set_index(month, drop=True).reindex(date_rng)[value]

    acc, month, value = df.columns.values
    date_rng = pd.date_range(start=min(df[month]), end=max(df[month]), freq='MS')
    df = df.groupby(acc).apply(dateSeries, date_rng=date_rng).T
    df.index.name = 'Time'
    df.to_csv(areaName)


def PlotUsageTimeseries(histo, title=''):
    histo = histo.iloc[:, histo.apply(lambda x: x.first_valid_index()).argsort()].fillna(-750)
    # histo['null_count'] = histo.isnull().sum(axis=1)
    # histo = histo.sort_values('null_count', ascending=False).drop('null_count', axis=1)

    fig = go.Figure(data=go.Heatmap(
        z=histo,
        y=histo.index,
        colorscale='earth',
        zmin=-1500, zmax=1500,
        reversescale=False
    ))
    fig.update_layout(title=title, xaxis_title="Users", yaxis_title="Time")
    fig.show()


def SeasonalBoxplot(alldf):
    monthlyAverage = alldf.groupby(alldf.index.strftime('%B')).mean().reindex(month_names, axis=0)

    fig = go.Figure()
    for i, row in monthlyAverage.iterrows(): fig.add_trace(
        go.Box(y=monthlyAverage.loc[i], name=i, marker_color=monthcolorDict[i], legendgroup=monthcolorDict[i]))
    fig.update_layout(title='Seasonal usage of electricity', yaxis_title="Unit",
                      xaxis_title="Month of Year", legend_orientation="h")
    fig.show()


if __name__ == '__main__':
    # allData = getAlldf(all_zones_path[:3]).dropna(how='all', axis=1)
    # PlotUsageTimeseries(allData)
    # print(allData)
    UsageTS(all_zones_path[:3])
