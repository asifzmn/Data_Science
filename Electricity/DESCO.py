import os
from collections import Counter

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from Electricity.ElecCommon import createAndSaveTimeSeries, processTS

mainpath = '/media/az/Study/Datasets/Electricity/DESCO data/'
tsPath, dlPath, rawPath, XBa2i, XCa2i = 'DESCO Data Timeseries (copy)/', 'DESCO Data Delay/', 'raw data/', 'XBa2i.csv', 'XCa2i.csv'
rawdataPath = mainpath + rawPath
zonesall = os.listdir(rawdataPath)
delList = ['Monipur', 'Kafrul', 'Agargaon', 'Tongi_West', 'Rupnagor', 'Shahali', 'Pallabi', 'Dakshinkhan'][:-1]
zones = list(filter(lambda x: x not in delList, zonesall))
month_names = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
               'November', 'December']
monthColors = ["#8fcadd"] * 2 + ["#46d246"] * 3 + ["#ff0000"] * 3 + ["#ffa500"] * 3 + ["#8fcadd"]
monthcolorDict = dict(zip(month_names, monthColors))
pal = ['#F70D1A', '#2B3856', '#C35817', '#B2C248', '#3BB9FF', '#893BFF', '#7D0541', '#F3E5AB', '#C12869', '#BCC6CC',
       '#8AFB17', '#348781', '#C7A317', '#EDE275', '#C6AEC7']


def mergecsv():
    fileChunk = ['/media/az/Study/Datasets/Electricity/DESCO data/Baridhara 18-6-2017/XBA2I',
                 '/media/az/Study/Datasets/Electricity/DESCO data/Gulshan/X B A2i du',
                 ]

    for fileDir in fileChunk:
        files = os.listdir(fileDir)
        df = pd.read_csv(fileDir + '/' + files[0])
        combined_csv = pd.concat([pd.read_csv(fileDir + '/' + f, header=None) for f in files[1:]])
        combined_csv.columns = df.columns
        combined_csv = pd.concat([df, combined_csv])
        combined_csv.to_csv(fileDir + ".csv", index=False)


def readCSVca(path, fName):
    def uniques(x): print(x.name, x.unique(), x.unique().shape, '\n')

    df = pd.read_csv(path + fName, encoding='cp1252')  # alternative "ISO-8859-1"
    df.apply(uniques)
    print(df.ZONE.value_counts())


# def UniqueValues(df):
#     df[['DUE_DATE', 'COLLECTION_DATE']] = df[['DUE_DATE', 'COLLECTION_DATE']].apply(pd.to_datetime)
#     print(sorted(Counter((df['DUE_DATE'] - df['COLLECTION_DATE']).dt.days.astype('int16')).keys()))
#     df.apply(uniques)


def PaymentTimingDistrict():
    # all = []
    # for zone in zones:
    #     df = processTS(dlpath + zone).fillna(0)
    #     print(df.shape)
    #     # print(df.where(df < 0).count().value_counts())
    #     all.append(df.apply(
    #         lambda x: 'Regular' if (x >= 0).all() else 'Late' if (x >= -30).all() else 'Skip').value_counts() /
    #                df.shape[1])
    #     # lambda x: 'Regular' if (x >= 0).all() else 'Late' if (x >= -30).all() else 'Skip').value_counts())
    #
    # concatDF = pd.concat(all, axis=1)
    # print(concatDF)
    # concatDF.columns = zones
    # print(concatDF)
    # concatDF = concatDF.loc[['Skip', 'Late', 'Regular']]
    # print(concatDF)
    # print(concatDF.mean(axis=1))

    # concatDF.to_csv('PaymentTimingDistrict')
    concatDF = pd.read_csv('Data Directory/PaymentTimingDistrict').set_index('Unnamed: 0')
    print(concatDF)

    fig = go.Figure(data=go.Heatmap(
        z=concatDF.values,
        y=concatDF.index,
        x=concatDF.columns,
        colorscale='Electric',
        reversescale=False, zmin=0, zmax=1
    ))
    fig.update_layout(font_size=18, title='Payment Delinquency Heatmap', xaxis_title="Zone",
                      yaxis_title="Payment Group")
    fig.show()


def PaymentTimingWithUsage():
    mainpath, ts, dl = '/home/az/PycharmProjects/ML/Electricity/Data Directory/', 'DESCO Data Timeseries/', 'DESCO Data Delay/'
    alldlq, allTs = [], []
    for zone in zones:
        print(zone)
        df = pd.read_csv(mainpath + dl + zone, index_col='ACC_NUMBER', low_memory=False).T.fillna(0)['2009':'2014']
        dlqncy = (df.apply(lambda x: 'Regular' if (x >= 0).all() else 'Late' if (x >= -30).all() else 'Skip'))
        alldlq.append(dlqncy)

        df = pd.read_csv(mainpath + ts + zone, index_col='ACC_NUMBER', low_memory=False).T['2009':'2014']
        bins = np.array([0, 100, 200, 300, 400, 500, 600, 10000])
        freqs = pd.cut(df.apply(lambda x: x.mean()), bins)
        allTs.append(freqs)

    dfmegred = (pd.concat([pd.concat(alldlq), pd.concat(allTs)], axis=1))
    dfmegred.to_csv('PaymentTimingWithUsage')
    dfmegred = pd.read_csv('Data Directory/PaymentTimingWithUsage', index_col='ACC_NUMBER')
    print(dfmegred.shape)

    cnt = pd.crosstab(dfmegred.iloc[:, 0], dfmegred.iloc[:, 1])
    cnt = cnt.loc[['Skip', 'Late', 'Regular']]
    cnt = cnt.apply(lambda x: x / x.sum())

    print(cnt)
    print(cnt.mean(axis=1))

    fig = go.Figure(data=go.Heatmap(
        z=cnt.values,
        y=cnt.index,
        x=['-'.join(x[1:-1].split(', ')) for x in cnt.columns],
        colorscale='Electric',
        reversescale=False, zmin=0, zmax=1
    ))
    fig.update_layout(font_size=18, title='Payment Delinquency Heatmap', xaxis_title="Unit",
                      yaxis_title="Payment Group")
    fig.show()


def LatePaymentDistrictUsage():
    mainpath, ts, dl = '/home/az/PycharmProjects/ML/Electricity/Data Directory/', 'DESCO Data Timeseries/', 'DESCO Data Delay/'
    # alldlq = []
    # for zone in zones:
    #     print(zone)
    #     df = pd.read_csv(mainpath + dl + zone, index_col='ACC_NUMBER', low_memory=False).T.fillna(0)['2009':'2014']
    #     dlqncy = (df.apply(lambda x: 'Regular' if (x >= 0).all() else 'Late' if (x >= -30).all() else 'Skip'))
    #
    #     df = pd.read_csv(mainpath + ts + zone, index_col='ACC_NUMBER', low_memory=False).T['2009':'2014']
    #     bins = np.array([0, 100, 200, 300, 400, 500, 600, 10000])
    #     freqs = pd.cut(df.apply(lambda x: x.mean()), bins)
    #
    #     dfmegred = (pd.concat([dlqncy, freqs], axis=1))
    #     cnt = pd.crosstab(dfmegred.iloc[:, 0], dfmegred.iloc[:, 1])
    #     cnt = cnt.loc[['Skip', 'Late']].sum()
    #     alldlq.append(cnt)
    #
    # cnt = (pd.concat(alldlq, axis=1))
    # cnt.columns = zones
    # cnt.to_csv('LatePaymentDistrictUsage')
    cnt = pd.read_csv('Data Directory/LatePaymentDistrictUsage').set_index('1').T
    cnt = cnt.apply(lambda x: x / x.sum(), axis=1)
    print(cnt)

    fig = go.Figure(data=go.Heatmap(
        z=cnt.values,
        y=cnt.index,
        x=['-'.join(x[1:-1].split(', ')) for x in cnt.columns],
        colorscale='Electric',
        reversescale=False,  # zmin=0, zmax=1
    ))
    fig.update_layout(font_size=18, title='Payment Delinquency Heatmap', xaxis_title="Unit",
                      yaxis_title="Zone", height=750, width=750)
    fig.show()


def daysDelayFrequency(df):
    delays = df['differ'].value_counts()

    fig = go.Figure(data=[
        go.Bar(y=delays, x=delays.index.values, marker_color="black")
    ])
    fig.show()


def getAlldf():
    dfs = [processTS(mainpath + tsPath + zone) for zone in zones]
    return pd.concat(dfs, axis=1)


def DataCleaning():
    for zone in os.listdir(mainpath + rawPath):
        cols = ['ACC_NUMBER', 'MONTH', 'YEAR', 'UNIT', 'COLLECTION_DATE', 'DUE_DATE']
        df = pd.read_csv(mainpath + rawPath + zone + '/' + XBa2i, usecols=cols, parse_dates=[4, 5]).dropna()
        df['Bill_Month'] = pd.to_datetime(df[['YEAR', 'MONTH']].assign(DAY=1))
        df['differ'] = (df['DUE_DATE'] - df['COLLECTION_DATE']).dt.days.astype('int16')
        df = df.drop(['MONTH', 'YEAR'], axis=1).drop_duplicates(subset=['ACC_NUMBER', 'Bill_Month'])
        createAndSaveTimeSeries(df[['ACC_NUMBER', 'Bill_Month', 'UNIT']], mainpath + tsPath + zone)
        createAndSaveTimeSeries(df[['ACC_NUMBER', 'Bill_Month', 'differ']], mainpath + dlPath + zone)


def SeasonalBoxplot(alldf):
    monthlyAverage = alldf.groupby(alldf.index.strftime('%B')).mean().reindex(month_names, axis=0)
    print(monthlyAverage)

    fig = go.Figure()
    for i, row in monthlyAverage.iterrows(): fig.add_trace(
        go.Box(y=monthlyAverage.loc[i], name=i, marker_color=monthcolorDict[i], legendgroup=monthcolorDict[i]))
    fig.update_layout(title='Seasonal usage of electricity', yaxis_title="Unit",
                      xaxis_title="Month of Year", legend_orientation="h")
    fig.show()


def UserGroups(df, title):
    bins = np.array([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 10000, 1000000])
    freqs = pd.cut(df.mean(), bins).value_counts().sort_index()
    freqs.name = title
    # freqsFrag = freqs/freqs.sum()
    return freqs


def UsageBarChart(usageGroup):
    # usageGroup.to_csv('usageGroup')
    # usageGroup = pd.read_csv('Data Directory/usageGroup', index_col=0)
    usageGroup = usageGroup.apply(lambda x: x / x.sum())

    rangeVal = ['-'.join(str(x)[1:-1].split(', ')) for x in usageGroup.index]

    fig = go.Figure(data=[
        go.Bar(name=d, x=rangeVal, y=usageGroup[d], marker_color=pal[i]) for i, d in enumerate(usageGroup)
        # go.Bar(name=d, x=rangeVal, y=usageGroup[d]) for i, d in enumerate(usageGroup)
    ])

    fig.update_layout(barmode='group', title='Electricity Usage Groups', yaxis_title="User Density",
                      xaxis_title="Units", legend_orientation="h")
    fig.show()


def FilterLTA(df, zone):
    mt = pd.read_csv(mainpath + rawPath + zone + '/' + XCa2i, encoding='cp1252').dropna()
    mtLTA = mt[(mt['CATEGORY'] == 'LT') & (mt['TARIFF'] == 'A')]['ACC_NUMBER'].astype(int).values.astype('str')
    return df.filter(items=mtLTA)


def TimeSeriesZone():
    fig = go.Figure()
    for i, zone in enumerate(zonesall):
        df = processTS(mainpath + tsPath + zone)
        df = FilterLTA(df, zone)
        # fig.add_trace(go.Scatter(x=df.index, y=df.mean(axis=1), mode='lines+markers', name=zone, marker_color=pal[i]))
        fig.add_trace(go.Scatter(x=df.index, y=df.mean(axis=1), mode='lines+markers', name=zone))

    fig.update_layout(font_size=18, title='Timeseries of average usage', xaxis_title="Time", yaxis_title="Unit")
    fig.show()


def ViolinPLot(usageGroup):
    usageGroup = usageGroup.apply(lambda x: x / x.sum())

    fig = go.Figure()

    for i, row in usageGroup.iterrows():
        fig.add_trace(go.Violin(y=row,
                                # name=day,
                                box_visible=True,
                                meanline_visible=True))

    fig.show()


if __name__ == '__main__':
    zones = zonesall[:]
    dfs = [pd.read_csv(mainpath + tsPath + zone).shape for zone in zones]
    print(dfs)
    exit()
    # MetaData()

    # prepareFromRawData()
    # df = pd.read_csv('/media/az/Study/Datasets/Electricity/DESCO data/DESCO Data Timeseries/Agargaon',index_col='UNIT',parse_dates=[0])
    # print(df)

    # TimeSeriesZone()

    # alldf = pd.concat([processTS(mainpath + tsPath + zone).mean(axis=1) for zone in zones], axis=1)
    # SeasonalBoxplot(alldf)

    usageGroup = pd.concat(
        [UserGroups(FilterLTA(processTS(mainpath + tsPath + zone), zone), zone) for i, zone in enumerate(zones)],
        axis=1)
    # usageGroup = pd.concat([UserGroups(processTS(mainpath + tsPath + zone),zone) for i, zone in enumerate(zonesall)],
    #                        axis=1)
    # UsageBarChart(usageGroup)
    ViolinPLot(usageGroup)

    # ProcessAll()
    # PaymentTimingDistrict()
    # PaymentTimingWithUsage()
    # LatePaymentDistrictUsage()

    # alldf = alldf.loc[:, (alldf < 10000).all(axis=0)]

    # duplicateRowsDF = df[df.duplicated(['ACC_NUMBER','MONTH','YEAR','DUE_DATE','COLLECTION_DATE'])]
    # duplicateRowsDF = df[df.duplicated()]
    # print(duplicateRowsDF)
    # duplicateRowsDF = df[df.duplicated(keep=False)]

    # duplicateRowsDF = df[df[['ACC_NUMBER','MONTH','YEAR','DUE_DATE','COLLECTION_DATE']].duplicated(keep='first')]
    # print(duplicateRowsDF)
    # duplicateRowsDF = df[df[['ACC_NUMBER','MONTH','YEAR','DUE_DATE','COLLECTION_DATE']].duplicated(keep=False)]

    # duplicateRowsDF = df[df[['ACC_NUMBER','MONTH','YEAR','DUE_DATE','COLLECTION_DATE']].duplicated(keep='first')]
    # print(duplicateRowsDF)
    # duplicateRowsDF = df[df[['ACC_NUMBER','MONTH','YEAR','DUE_DATE','COLLECTION_DATE']].duplicated(keep=False)]

    exit()
