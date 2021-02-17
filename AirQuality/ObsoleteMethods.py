import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter
from itertools import combinations
from matplotlib import pyplot as plt, cm
from plotly import express as px
from plotly import graph_objects as go
from sklearn.cluster import Birch, KMeans
from sklearn.preprocessing import MinMaxScaler
from AirQuality.DataPreparation import getCategoryInfo, LoadData
from AirQuality.Related.GeoMapMatplotLib import MapPlotting
from AirQuality.Related.Heatmap import heatmap, annotate_heatmap

cmap = ['Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap',
        'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd',
        'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2',
        'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r',
        'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r',
        'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral',
        'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd',
        'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg',
        'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper',
        'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray',
        'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r',
        'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r',
        'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'icefire', 'icefire_r', 'inferno', 'inferno_r', 'jet',
        'jet_r', 'magma', 'magma_r', 'mako', 'mako_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r',
        'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'rocket', 'rocket_r',
        'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r',
        'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'twilight', 'twilight_r',
        'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'vlag', 'vlag_r', 'winter', 'winter_r']

cmapPlotly = ['aggrnyl', 'agsunset', 'algae', 'amp', 'armyrose', 'balance',
              'blackbody', 'bluered', 'blues', 'blugrn', 'bluyl', 'brbg',
              'brwnyl', 'bugn', 'bupu', 'burg', 'burgyl', 'cividis', 'curl',
              'darkmint', 'deep', 'delta', 'dense', 'earth', 'edge', 'electric',
              'emrld', 'fall', 'geyser', 'gnbu', 'gray', 'greens', 'greys',
              'haline', 'hot', 'hsv', 'ice', 'icefire', 'inferno', 'jet',
              'magenta', 'magma', 'matter', 'mint', 'mrybm', 'mygbm', 'oranges',
              'orrd', 'oryel', 'peach', 'phase', 'picnic', 'pinkyl', 'piyg',
              'plasma', 'plotly3', 'portland', 'prgn', 'pubu', 'pubugn', 'puor',
              'purd', 'purp', 'purples', 'purpor', 'rainbow', 'rdbu', 'rdgy',
              'rdpu', 'rdylbu', 'rdylgn', 'redor', 'reds', 'solar', 'spectral',
              'speed', 'sunset', 'sunsetdark', 'teal', 'tealgrn', 'tealrose',
              'tempo', 'temps', 'thermal', 'tropic', 'turbid', 'twilight',
              'viridis', 'ylgn', 'ylgnbu', 'ylorbr', 'ylorrd']


def Plotting(Y, labels=None, xlabel='', title='', x=None):
    if x is None: x = np.arange(len(Y[0]))
    if labels is None: labels = np.arange(len(Y)) + 1
    # for y, label in zip((Y), labels): plt.plot(x, y, label=label)
    for y, label in zip((Y), labels): plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('pm2.5 Concentration')
    plt.legend()
    plt.show()


def HeatMap(data, tag, cbarlabel=None, precision=2, cmap="Purples"):
    fig, ax = plt.subplots()

    im, cbar = heatmap(data, tag, tag, ax=ax, cmap=cmap, cbarlabel=cbarlabel)
    _ = annotate_heatmap(im, valfmt="{x:." + str(precision) + "f} ")

    fig.tight_layout()
    plt.show()


def SeasonAnalysis(dataSummary, allDistrictMetaData):
    seasonNames = ['Summer', 'Rain', 'Winter']
    mappedReading = (np.vstack((np.transpose(dataSummary[0])[0].astype('float64'), dataSummary[1])))
    seasonData = [(mappedReading[1:, (np.isin(mappedReading[0], seasonMonth))]) for seasonMonth in
                  [np.arange(i * 4 + 2, i * 4 + 6) % 12 + 1 for i in range(3)]]
    seasonMean = np.transpose(np.mean(seasonData, axis=2))
    Plotting(seasonMean, x=seasonNames, labels=allDistrictMetaData[:, 0])


def DivisionAnalysis(dataSummaries, allDistrictMetaData):
    uniqueDivision = np.unique(allDistrictMetaData[:, 1])
    divisionMeans = [np.array([np.mean(dataSummary[1][allDistrictMetaData[:, 1] == x], axis=0) for x in uniqueDivision])
                     for dataSummary in dataSummaries]
    for divisionMean in divisionMeans: Plotting(divisionMean, uniqueDivision)


def Clustering(dataSummaries, allDistrictMetaData):
    nClusters = 9
    brc = Birch(n_clusters=nClusters)
    brc.fit(dataSummaries[0][1])
    arr = brc.predict(dataSummaries[0][1])
    print(arr)
    print([[np.where(arr == u)] for u in range(nClusters)])
    print([(allDistrictMetaData[:, 0, ][np.where(arr == u)]) for u in range(nClusters)])


def correlation(data, districtNames, timeParam):
    # shifting = 15
    # interpolated = (data)[:, -72:-48]
    shifting, interpolated = 27, (data)[:, -60:]
    timeLagMatrix = np.full((np.shape(interpolated)[0], np.shape(interpolated)[0]), -1)

    for comb in list(combinations(range(len(interpolated)), 2)):
        timeLag = (np.argmax([np.corrcoef(interpolated[comb[0], i:-shifting + i],
                                          interpolated[comb[1], int(shifting / 2):-int(shifting / 2) - 1])[0, 1] for i
                              in range(shifting)]) - int(shifting / 2))
        timeLagMatrix[comb[0], comb[1]], timeLagMatrix[comb[1], comb[0]] = timeLag, -timeLag
        timeLagMatrix[comb[0], comb[0]], timeLagMatrix[comb[1], comb[1]] = 0, 0

    HeatMap(timeLagMatrix, districtNames, precision=0, cmap='bwr')


def ClusteringSeries(df, timestamp):
    ls = ((df.values.flatten()))
    XX, YY = np.meshgrid(np.arange(df.shape[1]), np.arange(df.shape[0]))
    t, idxs = (np.vstack((ls, YY.ravel())).T), np.vstack((YY.ravel(), XX.ravel())).T
    # ls = MinMaxScaler(feature_range=(0, 1)).fit_transform(ls.reshape(-1, 1))
    # yy = MinMaxScaler(feature_range=(0, 1)).fit_transform(YY.ravel().reshape(-1, 1))
    # t,idxs = (np.vstack((ls.T, yy.T)).T),np.vstack((YY.ravel(),XX.ravel())).T
    # table = t
    table = MinMaxScaler(feature_range=(0, 1)).fit_transform(t)
    # print(table)

    name = 'clust'
    try:
        with open(name, "rb") as f:
            labels = pickle.load(f)
    except:
        nClusters = 10

        alg = KMeans(n_clusters=nClusters)
        alg.fit(table)
        labels = alg.labels_

        # clust = OPTICS(min_samples=5, xi=.001, min_cluster_size=35, p=1.5)
        # clust.fit(table)
        # labels = clust.labels_[clust.ordering_]

        #
        # brc = Birch(n_clusters=nClusters,threshold=.0001)
        # brc.fit(table)
        # labels = brc.predict(table)

        with open(name, "wb") as f:
            pickle.dump(labels, f)

    # print(len(np.unique(labels)))
    sns.set(style='darkgrid')
    colorGen = cm.get_cmap('jet', 256)
    pal = colorGen(np.linspace(0, 1, len(np.unique(labels))))[::-1]
    print(np.shape(labels))
    labels = [pal[l] for l in labels]
    print(np.shape(labels))
    # print((labels.reshape(df.shape)))
    plt.figure(figsize=(9, 6))
    print(np.shape(t))
    print(np.shape(table))
    print(np.shape(labels))
    plt.scatter(t[:, 1], t[:, 0], marker='o', s=15, linewidths=3, c=labels)

    # num_classes = 4
    # ts = range(10)
    # df = pd.DataFrame(data={'TOTAL': np.random.rand(len(ts)), 'Label': np.random.randint(0, num_classes, len(ts))},
    #                   index=ts)
    # print(df)

    # cmap = ListedColormap(['r', 'g', 'b', 'y'])
    # norm = BoundaryNorm(range(num_classes + 1), cmap.N)
    # points = np.array([df.index, df['TOTAL']]).T.reshape(-1, 1, 2)
    # segments = np.concatenate([points[:-1], points[1:]], axis=1)
    #
    # lc = LineCollection(segments, cmap=cmap, norm=norm)
    # lc.set_array(df['Label'])
    #
    # fig1 = plt.figure()
    # plt.gca().add_collection(lc)
    # plt.xlim(df.index.min(), df.index.max())
    # plt.ylim(-1.1, 1.1)
    # plt.show()

    # plt.scatter(t[:, 1], t[:, 0], marker='o', s=15, linewidths=3, c=labels)
    # for i in range(22):df.iloc[:,i].resample('D').max().plot(marker='o', markersize=1, linestyle=':',linewidth=0)

    plt.title(
        'Clustering on the max day value time series for month ' + str(timestamp.year) + ' ' + timestamp.strftime(
            '%B'))
    plt.xlabel('Monthday')
    plt.ylabel('PM 2.5 Reading')
    # plt.legend(np.unique(labels))
    # print(df.values)
    # df.plot(style='.')
    plt.show()

    # Clustering(reading.resample('D').max(), timeStamp)


def stringtoDftime(dataSummaries, TimeFrame, metaFrame):
    return pd.DataFrame(data=np.transpose(dataSummaries[1]), index=TimeFrame[0].values,
                        columns=[d.replace(' ', '') for d in metaFrame['Zone'].values]).apply(pd.to_numeric)[
           '2017':'2019']


def TimeSeries(df):
    pack = 1
    for i in range(0, 21, pack):
        df1 = df[1].iloc[:, i:i + pack]
        sns.set(style='darkgrid')
        df1.plot(alpha=0.9, style='.')
        timeDelta = 'W'
        df1.resample(timeDelta).mean().plot(style=':')
        # df.asfreq(timeDelta).plot(style='--')
        # plt.legend(df1.columns.values,loc='upper left')
        plt.legend([], loc='upper left')
        # plt.legend(['input', 'resample', 'asfreq'],loc='upper left')
        plt.show()


def resampledSeries(df):
    # df = df.iloc[24600:,4]
    sns.set(style='darkgrid')
    plt.figure(figsize=(15, 10))
    # sns.scatterplot(data=df.iloc[:,:4])
    # df.plot(linestyle=':', linewidth=1)
    # df.plot(style='.')
    for i in range(22):
        df.iloc[:, i].resample('D').max().plot(marker='o', markersize=1, linestyle=':', linewidth=0)
        # df.iloc[25000:,i].plot(style='.')
    # df.resample('2H').mean().plot(marker='o', markersize=3, linestyle='--',linewidth=1)
    # df.resample('3H').mean().plot(marker='s', markersize=3, linestyle='-',linewidth=1)
    # df.resample('D').mean().plot(marker='o', markersize=1, linestyle='-',linewidth=5)
    # df.resample('D').max().plot(marker='o', markersize=1, linestyle=':',linewidth=0)
    # df.resample('3D').mean().plot(marker='s', markersize=1, linestyle='-',linewidth=5)
    # df.resample('W').mean().plot(marker='h', markersize=1, linestyle='-',linewidth=5)
    # df.resample('3W').mean().plot(marker='x', markersize=1, linestyle='-',linewidth=5)
    # df.resample('M').mean().plot(marker='d', markersize=15, linestyle='-',linewidth=5)
    # df.resample('3M').mean().plot(marker='v', markersize=15, linestyle='-',linewidth=5)
    # plt.savefig('ser',dpi=300)
    # plt.legend("")
    plt.show()


def Scapping(driver):
    element = driver.find_element_by_xpath(
        '// *[ @ id = "wrapper-main"] / div / main / div / div[2] / form / div[4] / div[4]')
    factors = np.array(element.find_elements_by_xpath(".//div"))
    for factor in factors[np.delete(np.arange(32 - 22), [0, 3, 15])]: factor.click()
    for tickBox in ["swrad", "directrad", "cape"]: driver.find_element_by_id(tickBox).click()


def Visualization(dataSummaries, basicTimeParameters, metaFrame):
    for datasummary, timeParam in zip(dataSummaries, basicTimeParameters):
        # Plotting(np.array(datasummary[1]), labels=allDistrictMetaData[:, 0],x=['-'.join(map(str, st)) for st in datasummary[0]], xlabel=timeParam)
        # Plotting(np.array(datasummary[1]),x=['-'.join(map(str, st)) for st in datasummary[0]], xlabel=timeParam)
        # HeatMap(np.corrcoef(np.array(datasummary[1])), allDistrictMetaData[:, 0],cbarlabel=timeParam)
        # correlation(datasummary[1],allDistrictMetaData[:, 0],timeParam)

        for i in range(16720, 16730):
            # if('/'.join(map(str,datasummary[0][i][::-1]))=='7/11/2018'):
            MapPlotting(metaFrame, datasummary[1][:, i],
                        ('/'.join(map(str, datasummary[0][i][-2::-1])) + ' hour ' + datasummary[0][i][-1]))


def GraphObjPlotly(metaFrame):
    fig = go.Figure()

    for idx, row in metaFrame.iterrows():
        fig.add_trace(go.Scattermapbox(
            mode="markers",
            lon=[row['Longitude']],
            lat=[row['Latitude']],
            name=row['labels'],
            # fillcolor = colorScale[0],
            marker={'size': 10, 'color': row['category']}))

    fig.add_trace(go.Scattermapbox(
        mode="lines",
        lon=metaFrame['Longitude'],
        lat=metaFrame['Latitude'],
        marker={'size': 10}))

    fig.update_layout(
        margin={'l': 0, 't': 0, 'b': 0, 'r': 0},
        mapbox={
            'center': {'lon': 90, 'lat': 23},
            'style': "stamen-terrain",
            # 'style': "open-street-map",
            'zoom': 7.5})
    fig.show()


def SliderMap(df):
    timeformat = 'M'
    dfre = df.resample(timeformat).mean()
    data, times = (dfre).stack().values, (dfre.index.strftime('%Y %B'))

    colorScale, categoryName, AQScale = getCategoryInfo()
    dataVector, metaFrame = LoadData(name='reading.pickle')

    metaFrame = pd.concat([metaFrame] * len(times), ignore_index=True)
    metaFrame['category'], metaFrame['time'] = [categoryName[val] for val in
                                                np.digitize(data, AQScale[1:-1])], np.repeat(times, df.shape[1])

    fig = px.scatter_mapbox(metaFrame, lat="Latitude", lon="Longitude", hover_name="Zone", animation_frame="time",
                            # hover_data=["Population"],
                            zoom=7.5, height=900, color="category", color_discrete_sequence=colorScale)

    # fig = px.line_mapbox(metaFrame, lat="Latitude", lon="Longitude", color="category", zoom=3, height=300)

    fig.update_traces(marker_size=18, marker_opacity=2 / 3)
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.update_layout(legend=dict(bordercolor='rgb(100,100,100)', borderwidth=2, x=0, y=0))
    fig.show()


def SliderMapHeat(df, metaFrame, ):
    timeformat = '3H'
    dfre = df.resample(timeformat).mean()
    data, times = (dfre).stack().values, (dfre.index.strftime('%Y-%B-%D  %H:00:00 '))

    colorScale, categoryName, AQScale = getCategoryInfo()

    metaFrame = pd.concat([metaFrame] * len(times), ignore_index=True)
    metaFrame['category'], metaFrame['time'] = data, np.repeat(times, df.shape[1])

    fig = px.scatter_mapbox(metaFrame, lat="Latitude", lon="Longitude", hover_name="Zone", animation_frame="time",
                            # hover_data=["Population"],
                            zoom=7.5, height=900, color="category", range_color=[10, 40],
                            color_continuous_scale=([(0, "yellow"), (0.5, "orange"), (1, "red")]))

    # fig = px.line_mapbox(metaFrame, lat="Latitude", lon="Longitude", color="category", zoom=3, height=300)

    fig.update_traces(marker_size=18, marker_opacity=2 / 3)
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.update_layout(legend=dict(bordercolor='rgb(100,100,100)', borderwidth=2, x=0, y=0))
    fig.show()


def Imputation():
    fig, ax = plt.subplots(2, sharex=True)
    data = df.iloc[:10, :2]
    data.asfreq('H').plot(ax=ax[0], marker='o')
    data.asfreq('H', method='bfill').plot(ax=ax[1], style='-o')
    data.asfreq('H', method='ffill').plot(ax=ax[1], style='--o')
    ax[1].legend(["back-fill", "forward-fill"])
    plt.show()


def SeriesBox(monthData, df):
    color = np.array(['royalblue'] * (len(monthData) - 6) + ['indianred'] * 6)[
        [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 24, 25, 26, 27, 28, 29]]

    for d in df:
        fig = go.Figure()
        for i, m in enumerate(monthData[[0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 24, 25, 26, 27, 28, 29]]):
            fig.add_trace(go.Box(y=m[2][d], name=str(m[1]), marker_color=color[i]))
            fig.update_layout(title=d)
        fig.show()


def missingFilter(df, metaFrame):
    missing = df.isnull().astype(int)
    bools = (df.isnull().sum() < 5055)
    bools[0] = True
    df = (df.loc[:, bools])
    metaFrame = metaFrame.loc[bools]


def dataInterpolation(matrixData):
    cutTime, cutData = matrixData
    solidDataLoc = (np.where(np.any(cutData == None, axis=0) == False))[0]
    if not len(solidDataLoc) == 0: cutTime, cutData = cutTime[solidDataLoc[0]:solidDataLoc[-1]], cutData[:,
                                                                                                 solidDataLoc[0]:
                                                                                                 solidDataLoc[-1]]
    df = pd.DataFrame.from_records(np.transpose(cutData))
    return cutTime, np.transpose(
        df.interpolate(method='pchip', order=5, limit_direction='forward', limit_area=None, axis=0).to_numpy())


def Bucketing(reading, bins):
    pm25conc = np.transpose(reading.to_numpy())
    inds = np.digitize(pm25conc, bins)

    binsC = np.array([np.hstack((counts, np.zeros(len(bins) + 1 - len(counts)))) for counts in
                      [np.bincount(inds[i]) for i in range(reading.shape[1])]])
    return binsC / binsC.sum(1, keepdims=True)

def LeaderFollower(df):
    t_start, tau, step_size, window_size = 0, 3, 72, 180
    t_end = t_start + window_size
    rss = []
    while t_end < df.shape[0]:
        d1 = df['Feni'].iloc[t_start:t_end]
        d2 = df['Rajshahi'].iloc[t_start:t_end]
        rs = [crosscorr(d1, d2, lag) for lag in range(-int(tau), int(tau + 1))]
        rss.append(rs)
        t_start += step_size
        t_end += step_size
    rss = pd.DataFrame(rss)

    f, ax = plt.subplots(figsize=(5, 20))
    sns.heatmap(rss, cmap=sns.diverging_palette(175, 250, s=90, n=27), ax=ax)
    ax.set(title='Rolling Windowed Time Lagged Cross Correlation', xlabel='Offset', ylabel='Epochs')
    # ax.set_xticks([0, 50, 100, 151, 201, 251, 301])
    # ax.set_xticklabels([-150, -100, -50, 0, 50, 100, 150]);
    plt.show()

def ShortLengthImputation(dataSummaries):
    dataset = dataSummaries[1]
    for i in range(22):
        idx_pairsY = np.where(np.diff(np.hstack(([False], dataset[i] is None, [False]))))[0].reshape(-1, 2)
        diffsY = idx_pairsY[:, 1] - idx_pairsY[:, 0]
        # print(diffsY)
        for gap in idx_pairsY[diffsY <= 3][:-1]: dataset[i][gap[0]:gap[1]] = np.linspace(dataset[i][gap[0] - 1],
                                                                                         dataset[i][gap[1]],
                                                                                         gap[1] - gap[0] + 2)[1:-1]
    dataSummaries = dataSummaries[0], dataset
    return dataSummaries


def weightedChoice():
    x = []
    x.append(Counter(np.random.choice(['H', 'T'], 10, p=[0.5, 0.5]))['H'] / 10)
    x.append((Counter(np.random.choice(['H', 'T'], 100, p=[0.5, 0.5]))['H'] / 100))
    x.append((Counter(np.random.choice(['H', 'T'], 1000, p=[0.5, 0.5]))['H'] / 1000))
    x.append((Counter(np.random.choice(['H', 'T'], 10000, p=[0.5, 0.5]))['H'] / 10000))
    x.append((Counter(np.random.choice(['H', 'T'], 100000, p=[0.5, 0.5]))['H'] / 100000))
    print(x)
