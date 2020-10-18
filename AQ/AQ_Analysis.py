import re
import geojsoncontour
from collections import Counter
from itertools import combinations
from matplotlib.colors import LinearSegmentedColormap
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from statsmodels.tsa.seasonal import seasonal_decompose
from AQ.Correlation_Measures import *
from AQ.DataPreparation import *
from AQ.GeoMapPlotly import SliderMapCommon
from AQ.GeoPandas import mapArrow, mapPlot
from AQ.MeteoblueInfoAnalysis import *
from AQ.Related.GeoMapMatplotLib import *
import plotly.graph_objects as go
import more_itertools
from pandas_profiling import ProfileReport

colorScale, categoryName, AQScale = getCategoryInfo()


def ExtremeCorrelation(df):
    corrDistrict = np.corrcoef(np.transpose(df.to_numpy()))
    correlatedDistricts = (np.transpose(np.where(corrDistrict > 0.999)))
    identity = np.transpose([np.arange(len(df)), np.arange(len(df))])
    closeIndexReal = (sorted(list(set(map(tuple, correlatedDistricts)).difference(set(map(tuple, identity))))))
    print(np.transpose(
        [df.columns.values[np.transpose(closeIndexReal)[0]], df.columns.values[np.transpose(closeIndexReal)[1]]]))


def SeasonalDecomposition(df):
    result = seasonal_decompose(df)
    print(result)
    result.plot()
    plt.show()

    # x = df - result.seasonal
    # diff = np.diff(x)
    # plt.plot(diff)
    # plt.show()
    #
    # plot_acf(diff, lags=10)
    # plot_pacf(diff, lags=10)
    # plt.show()


def LatexTableFormat(df):
    new_df = df.groupby([df.index.year]).agg(['min', 'mean', 'max'])
    new_df.index.set_names(['Year'], inplace=True)
    print(new_df.T.to_latex(col_space=3))


def makeHeaderFile(metaFrame):
    metaFrame = metaFrame.assign(Country='Bangladesh')
    metaFrame = metaFrame.reset_index()[['Country', 'Division', 'Zone']]
    metaFrame.to_csv('headerFile.csv', index=False)


def BoxPlotSeason(df):
    monthColors = ["#8fcadd"] * 2 + ["#46d246"] * 3 + ["#ff0000"] * 3 + ["#ffa500"] * 3 + ["#8fcadd"]
    seasons = ['Spring', 'Winter', 'Summer', 'Autumn']
    seasonPalette = dict(zip((np.unique(df.index.month)), monthColors))
    df = df.resample('6H').mean()

    for district in df.columns.values[:]:
        ax = sns.boxplot(x=df.index.month, y=district, data=df, palette=seasonPalette)  # weekday_name,month,day,hour
        for seasonName, color in zip(seasons, np.unique(monthColors)): plt.scatter([], [], c=color, alpha=0.66, s=150,
                                                                                   label=str(seasonName))
        plt.legend(scatterpoints=1, frameon=False, labelspacing=.5, title='Season')
        pltSetUpAx(ax, xlabel="Month", ylabel="PM Reading", title=district, ylim=(0, 300))


def BoxPlotHour(df):
    cm = LinearSegmentedColormap.from_list('DayNight',
                                           ["#075077", "#075077", "#6f5a66", "#6f5a66",
                                            "#eae466", "#eae466", "#eae466", "#eae466",
                                            "#6f5a66", "#6f5a66", "#075077", "#075077"], N=24)
    hourPalette = dict(zip((np.unique(df.index.hour)), cm(np.arange(24))))
    for district in df.columns.values[:]:
        ax = sns.boxplot(x=df.index.hour, y=district, data=df, palette=hourPalette)  # weekday_name,month,day,hour
        # pltSetUpAx(ax, "Hour of Day", "PM Reading", district + ' in ' + str(timeStamp), ylim=(0, 225))
        pltSetUpAx(ax, xlabel="Hour of day", ylabel="PM Reading", title=district, ylim=(0, 450))


def PairDistributionSummary(df, samplingHours=1):
    df = df[: str(max(df.index.date) + timedelta(days=-1))].resample(str(samplingHours) + 'H').mean()
    df['daytime'] = np.tile(
        np.hstack((np.repeat('Day', 12 // samplingHours), np.repeat('Night', 12 // samplingHours))),
        (df.shape[0] // 24))
    df = df.sort_values(by=['daytime'], ascending=False)
    g = sns.pairplot(df, hue='daytime', palette=["DARKBLUE", "BURLYWOOD"], plot_kws={"s": 9})

    for ax in plt.gcf().axes: ax.set_xlabel(ax.get_xlabel(), fontsize=30)
    for ax in plt.gcf().axes: ax.set_ylabel(ax.get_ylabel(), fontsize=30)

    g.fig.get_children()[-1].set_bbox_to_anchor((1.1, 0.5, 0, 0))
    plt.show()


def pltSetUp(xlabel=None, ylabel=None, title=None, xlim=None, ylim=None, save=None):
    if not xlabel is None: plt.xlabel(xlabel)
    if not ylabel is None: plt.ylabel(ylabel)
    if not title is None: plt.title(title)
    if not xlim is None: plt.xlim(xlim[0], xlim[1])
    if not ylim is None: plt.ylim(ylim[0], ylim[1])

    if save is None:
        plt.show()
    else:
        plt.savefig(save + '/' + title + '.png', dpi=300)
        plt.clf()


def pltSetUpAx(ax, xlabel=None, ylabel=None, title=None, xlim=None, ylim=None, save=None):
    if not xlabel is None: ax.set_xlabel(xlabel)
    if not ylabel is None: ax.set_ylabel(ylabel)
    if not title is None: ax.set_title(title)
    if not xlim is None: ax.set_xlim(xlim[0], xlim[1])
    if not ylim is None: ax.set_ylim(ylim[0], ylim[1])

    if save is None:
        plt.show()
    else:
        plt.savefig(save + '/' + title + '.png', dpi=300)
        plt.clf()


def HeatMapDriver(data, vmin, vmax, title, cmap):
    mask = np.zeros_like(data, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        ax = sns.heatmap(data, mask=mask, cmap=cmap, square=True, linewidths=.5, vmin=vmin,
                         vmax=vmax, xticklabels=True, yticklabels=True)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    ax.set_title(title)
    plt.show()


def KDE(df):
    for comb in list(combinations(range(len(df.columns.values)), 2)):
        with sns.axes_style('white'): sns.jointplot(df[df.columns.values[comb[0]]], df[df.columns.values[comb[1]]],
                                                    data=df, kind='kde')
        plt.show()


def TriangularHeatmap(title, df):
    plt.figure(figsize=(6, 6))
    HeatMapDriver(df.corr(), 0, 1, str(title.year) + '/' + str(title.month), 'Purples')


def PlotlyRosePlotBasic(info=None):
    if info is None:
        info = [[[77.5, 72.5, 70.0, 45.0, 22.5, 42.5, 40.0, 62.5], '11-14 m/s', 'rgb(106,81,163)'],
                [[55.5, 50.0, 45.0, 35.0, 20.0, 22.5, 37.5, 55.0], '8-11 m/s', 'rgb(158,154,200)'],
                [[40.0, 30.0, 30.0, 35.0, 7.5, 7.5, 32.5, 40.0], '5-8 m/s', 'rgb(203,201,226)'],
                [[20.0, 7.5, 15.0, 22.5, 2.5, 2.5, 12.5, 22.5], '< 5 m/s', 'rgb(242,240,247)']
                ]

    fig = go.Figure()
    for [r, name, marker_color] in info:
        fig.add_trace(go.Barpolar(
            r=r,
            name=name,
            marker_color=marker_color
        ))

    fig.update_traces(
        text=['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'])
    # fig.update_traces(text=['North', 'N-E', 'East', 'S-E', 'South', 'S-W', 'West', 'N-W'])
    fig.update_layout(
        title='Wind Speed Distribution in Laurel, NE',
        font_size=16,
        legend_font_size=16,
        polar_radialaxis_ticksuffix='%',
        polar_angularaxis_rotation=90,
        template="plotly_dark",
        polar_angularaxis_direction="clockwise"
    )
    fig.show()


def PLotlyTimeSeries(df, missing=None):
    fig = go.Figure()
    # fig.add_trace(go.Scatter(x=df.index, y=df['Dhaka'], name="Dhaka",line_color='deepskyblue'))
    # fig.add_trace(go.Scatter(x=df.Date, y=df['AAPL.Low'], name="AAPL Low",line_color='dimgray'))
    for d in df: fig.add_trace(go.Scatter(x=df.index, y=df[d], name=d))
    # for d in df.columns.values[:]: fig.add_trace(go.Scatter(x=df.index, y=df[d], name=d, marker=dict(color=missing[d])))

    fig.update_traces(mode='markers+lines', marker=dict(line_width=0, symbol='circle', size=5))
    # fig.update_layout(title_text='Time Series with Rangeslider',xaxis_rangeslider_visible=True)
    fig.show()


def Bucketing(reading, bins): return reading.apply(
    lambda x: pd.cut(x, bins).value_counts().sort_index()).apply(
    lambda x: x / x.sum())


def ratioMapPlotting(reading):
    norm = Bucketing(reading, [55.5, 500])
    MapPlotting(metaFrame, df.mean().values, ratiodata=norm, title=str(timeStamp.year))


def LeaderFollower():
    df = reading
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


def FrequencyClustering(df):
    metaFrame = LoadMetadata()
    norm = Bucketing(df, AQScale).T
    dataPoints, n_clusters = norm.values[:, :], 6
    alg = KMeans(n_clusters=n_clusters)
    alg.fit(dataPoints)

    labels = alg.labels_
    labels = pd.DataFrame(labels, index=metaFrame.index, columns=['label'])

    # districtMeanandLabels = pd.concat([df.mean(), labels], axis=1)
    districtMeanandLabels = labels.assign(mean=df.mean())

    districtMeanAndGroup = districtMeanandLabels.groupby('label').mean().sort_values('mean').round(2).assign(
        color=['#C38EC7', '#3BB9FF', '#8AFB17', '#EAC117', '#F70D1A', '#7D0541'][:n_clusters])
    districtMeanAndGroup.columns = ['category', 'color']

    labels['color'] = labels.apply(lambda x: districtMeanAndGroup.loc[x['label']]['color'], axis=1)
    # labels['color'] = '#566D7E'

    metaFrame = pd.concat([metaFrame, labels], axis=1)

    representative = districtMeanandLabels.groupby('label').apply(lambda x: x['mean'].idxmax())[
        districtMeanAndGroup.index]
    print(representative)

    # mapPlot(metaFrame,('Average Reading',districtMeanAndGroup))

    # BoxPlotHour(df[representative])
    # BoxPlotSeason(df[representative])
    # PairDistributionSummary(df[representative])
    # PLotlyTimeSeries(df['2019-09':'2019-12'][representative])


def MissingDataInfo(df):
    print((df.isna().sum()))
    print(df.isnull().any(axis=1).sum())  # any null reading of a district for a time stamp
    print(df.isnull().all(axis=1).sum())  # all null reading of a district for a time stamp
    # print(df.notnull().any(axis=1).sum())
    # print(df.notnull().all(axis=1).sum())

    # HeatMapDriver(dfm.corr(), -1, 1, '', 'RdBu')


def MissingDataHeatmap(df):
    missing = df.T.isnull().astype(int)
    print(df.isnull().sum())

    bvals, dcolorsc = np.array([0, .5, 1]), []
    tickvals = [np.mean(bvals[k:k + 2]) for k in range(len(bvals) - 1)]
    ticktext, colors = ['Present', 'Missing'], ['#C6DEFF', '#2B3856']

    nvals = [(v - bvals[0]) / (bvals[-1] - bvals[0]) for v in bvals]  # normalized values
    for k in range(len(colors)): dcolorsc.extend([[nvals[k], colors[k]], [nvals[k + 1], colors[k]]])

    fig = go.Figure(data=go.Heatmap(
        z=missing.values,
        y=missing.index.values,
        x=pd.Series(df.index),
        colorscale=dcolorsc,
        colorbar=dict(thickness=75, tickvals=tickvals, ticktext=ticktext),
    ))

    fig.update_layout(
        title="Missing Data Information",
        yaxis_title="District",
        xaxis_title="Days",
        font=dict(
            size=9,
            color="#3D3C3A"
        )
    )
    fig.show()


def MissingBar(df):
    singleDistrictData = df['Dhaka'].values
    idx_pairs = np.where(np.diff(np.hstack(([False], (np.isnan(singleDistrictData)), [False]))))[0].reshape(-1, 2)
    counts = (Counter(idx_pairs[:, 1] - idx_pairs[:, 0]))
    sortedCounts = dict(sorted(counts.items()))

    # x = ['One', 'Two', 'Three', 'Four', 'More']
    # y = list(sortedCounts.values())[:4] + [sum(list((sortedCounts.values()))[4:])]

    y = list(sortedCounts.values())
    x = list(sortedCounts.keys())
    # x = np.arange(len(y))+1

    fig = go.Figure(data=[go.Bar(x=x, y=y)])
    fig.update_traces(marker_color='#3090C7', marker_line_color='#2554C7', marker_line_width=1.2, opacity=0.8)
    fig.update_layout(title_text='Missing Value Length')
    fig.show()


def ShiftedSeries(df, dis, lagRange, offset, rs):
    plt.figure(figsize=(9, 3))
    df[dis[0]].plot(linestyle='-', linewidth=1, c="Blue")
    df[dis[1]].plot(linestyle='-', linewidth=1, c="Red")
    df[dis[1]].shift(offset).plot(linestyle=':', linewidth=3, c="palevioletred")
    plt.xlabel('Time')
    plt.ylabel('PM2.5 Concentration')
    plt.legend([dis[0], dis[1], dis[1] + " Shifted"],
               loc='upper left')

    # loc = 'shiftedSeries/'+df.columns.values[comb[0]]+df.columns.values[comb[1]]+'Reading.png'
    # plt.savefig(loc,dpi=300)
    # plt.clf()
    plt.show()

    f, ax = plt.subplots(figsize=(9, 3))
    ax.plot(list(range(-lagRange, lagRange + 1)), rs, marker='o', )
    # ax.axvline(np.ceil(len(rs) / 2), color='k', linestyle='--', label='Center')
    # ax.axvline(np.argmax(rs), color='r', linestyle='--', label='Peak synchrony')
    ax.axvline(offset, color='k', linestyle='--', label='Peak synchrony')
    ax.set(title='Offset for {dis[0]} to {dis[1]} = {offset} hours',
           ylim=[-1, 1], xlabel='Offset', ylabel='Pearson r')
    plt.legend()

    # loc = 'shiftedSeries/' + df.columns.values[comb[0]] + df.columns.values[comb[1]] + 'Offset.png'
    # plt.savefig(loc, dpi=300)
    # plt.clf()
    plt.show()


def crosscorr(datax, datay, lag=0, wrap=False): return datax.astype('float64').corr(datay.shift(lag).astype('float64'))


def crosscorrBalanced(datax, datay, lag=0, wrap=False):
    return datax.astype('float64').corr(datay.shift(lag).astype('float64'))


def CrossCorr(timeDel, timeStamp, df, lagRange=3):
    # r_window_size = 24
    #
    # rolling_r = df.iloc[:, 0].rolling(window=r_window_size, center=True).corr(df.iloc[:, 1])
    # f, ax = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    # df.iloc[:, 0:2].rolling(window=r_window_size, center=True).median().plot(ax=ax[0])
    # ax[0].set(xlabel='Frame', ylabel='Series')
    # rolling_r.plot(ax=ax[1])
    # ax[1].set(xlabel='Frame', ylabel='Pearson r')
    # plt.suptitle("Rolling window correlation")
    # plt.show()
    # print(list(combinations(range(len(df.shape)), 2)))

    lags, lagDir, lagmatrix = [], [], np.zeros((df.shape[1], df.shape[1]))
    for comb in list(combinations(range((df.shape[1])), 2)):
        rs = [crosscorr(df[df.columns.values[comb[0]]], df[df.columns.values[comb[1]]], lag) for lag in
              range(-lagRange, lagRange + 1)]
        offset = - int(np.floor(len(rs) / 2) - np.argmax(rs))
        lags.append(offset)
        lagmatrix[comb[1]][comb[0]], lagmatrix[comb[0]][comb[1]] = offset, -offset

        if offset == 0: continue

        # print(df.columns.values[comb[0]],df.columns.values[comb[1]],offset)
        # ShiftedSeries(df,[df.columns.values[comb[0]],df.columns.values[comb[1]]],lagRange,offset,rs)

        # if offset < 0:
        #     dir = (angleFromCoordinate(df.columns.values[comb[0]], df.columns.values[comb[1]]))
        # else:
        #     dir = (angleFromCoordinate(df.columns.values[comb[1]], df.columns.values[comb[0]]))
        # lagDir.append(dir)

    # if len(lagDir)>0:
    #     WindGraphTeamEstimate(np.array(lagDir), ['Overall'])
    #     st = ((timeStamp.to_pydatetime()))
    #     en = ((timeStamp.to_pydatetime())+timedelta(hours=48))
    #     WindGraphTeam(dfm.loc[:,st:en,:])
    # print(df.columns.values[comb[0]],df.columns.values[comb[1]],offset)

    if list(Counter(lags).keys()) == [0]: return lagmatrix, lagDir

    mat = pd.DataFrame(data=lagmatrix.astype('int'), columns=df.columns.values, index=df.columns.values)
    # print(mat.dtypes)
    # HeatMapDriver(mat, -lagRange, lagRange, str(timeStamp),
    #               sns.diverging_palette(15, 250, s=90, l=50, n=90, center="light"))
    # MapPlotting(metaFrame, df.mean().values, vec=lagmatrix, title=str(timeStamp))
    print(Counter(lags))
    mapArrow(np.zeros(metaFrame.shape[0]).astype('int'), mat, df.index.date)

    return lagmatrix, lagDir


def WindGraphTeamEstimate(meteoData, alldis):
    colorPal = np.array(['#ffffff'])
    directions = np.array([[[getSides(meteoData), 'Wind']]])
    PlotlyRosePlot(directions, colorPal, alldis)


def MeteoAnalysis(df):
    meteoData = xr.open_dataset('meteoData.nc')['meteo']

    factors, districts = pd.Series(
        ['Temperature [2 m]', 'Relative Humidity [2 m]', 'Mean Sea Level Pressure', 'Precipitation',
         'Cloud Cover High', 'Cloud Cover Medium', 'Cloud Cover Low', 'Sunshine Duration', 'Shortwave Radiation',
         'Direct Shortwave Radiation', 'Diffuse Shortwave Radiation', 'Wind Gust', 'Wind Speed [10 m]',
         'Wind Direction [10 m]', 'Wind Speed [80 m]', 'Wind Direction [80 m]', 'Wind Speed [900 mb]',
         'Wind Direction [900 mb]', 'Wind Speed [850 mb]', 'Wind Direction [850 mb]', 'Wind Speed [700 mb]',
         'Wind Direction [700 mb]', 'Wind Speed [500 mb]', 'Wind Direction [500 mb]', 'Temperature [1000 mb]',
         'Temperature [850 mb]', 'Temperature [700 mb]', 'Surface Temperature', 'Soil Temperature [0-10 cm down]',
         'Soil Moisture [0-10 cm down]']), df.columns.values

    for lag in range(0, 3, 3):
        z = factors.apply(lambda x: df.shift(lag).corrwith(getFactorData(meteoData, x), axis=0))
        z.index = factors.values

        print(z)

        fig = go.Figure(data=go.Heatmap(
            z=z,
            y=factors,
            x=districts,
            colorscale='icefire',
            zmin=-1, zmax=1,
            reversescale=True
        ))
        fig.update_layout(
            autosize=False, width=1800, height=450*3,
            title="PM2.5 correaltion with meteorological factors",
            xaxis_title="District", yaxis_title="Factors",
            font=dict(size=21, color="#3D3C3A"
                      )
        )
        fig.show(config={'displayModeBar': False, 'responsive': True})


def FillMissingDataFromHours(x, hours=1):
    ss = [x.shift(shft, freq='H') for shft in np.delete(np.arange(-hours, hours + 1), hours)]
    return x.fillna((pd.concat(ss, axis=1).mean(axis=1)))


def FillMissingDataFromDays(x, days=3):
    ss = [x.shift(shft, freq='D') for shft in np.delete(np.arange(-days, days + 1), days)]
    return x.fillna((pd.concat(ss, axis=1).mean(axis=1)))


def FillMissingDataFromYears(y):
    ss = [y.shift(shft, freq='D') for shft in [-365 * 2, -365, 365, 365 * 2]]
    # ss = [x.shift(shft, freq='D') for shft in [-365, 365]]
    return y.fillna((pd.concat(ss, axis=1).mean(axis=1)))


def CorrationSeasonal(corrArray, rows=2, cols=2, title=''):
    subpTitles = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
                  'November', 'December']
    # subpTitles = ['Winter', 'Spring', 'Summer', 'Autumn']
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=subpTitles, vertical_spacing=0.1)
    # fig = fig.update_yaxes(side='right')

    import plotly.figure_factory as ff

    annotList = []

    for i in range(rows):
        for j in range(cols):
            fig1 = ff.create_annotated_heatmap(z=np.flip(np.tril((corrArray[i][j])), 0),
                                               y=metaFrame.index.to_list()[::-1], x=metaFrame.index.to_list(),
                                               annotation_text=np.flip(np.tril(
                                                   np.absolute(np.random.randint(3, size=(df.shape[1], df.shape[1])))),
                                                   0), colorscale='purpor', zmin=0, zmax=1)
            fig.add_trace(fig1.dataset[0], i + 1, j + 1)

            annot = list(fig1.layout.annotations)
            # annotList.extend(annot)

            for k in range(len(annot)):
                annot[k]['xref'] = 'x' + str(i * rows + j + 1)
                annot[k]['yref'] = 'y' + str(i * rows + j + 1)

    fig.update_layout(annotations=annotList)

    # for i in range(len(fig.layout.annotations)):
    #     fig.layout.annotations[i].font.size = 9

    for i, t in enumerate(subpTitles):
        fig['layout']['annotations'][i].update(text=t, font=dict(
            family="Courier New, monospace",
            size=27,
            color="#7f7f7f"
        ))

    fig.update_layout(title_text=title, height=7500, width=1500)

    fig.show()


def BoxPlotYear(df):
    for district in df.columns.values[:]:
        ax = sns.boxplot(x=df.index.year, y=district, data=df, color="#00AAFF")  # weekday_name,month,day,hour
        pltSetUpAx(ax, "Year", "PM Reading", 'Yearly average reading in ' + district, ylim=(0, 450))


def BoxPlotDistrict(df):
    plt.figure(figsize=(20, 8))
    ax = sns.boxplot(data=df, color="grey")
    for i, c in enumerate(colorScale): ax.axhspan(AQScale[i], AQScale[i + 1], facecolor=c, alpha=0.3)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", size=21)
    ax.set(ylim=(0, 250))
    plt.show()


def BoxPlotWeek(df):
    weekDay = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    dayColors = ["#ff0000"] * 2 + ["#8fcadd"] * 5
    seasons = ['Weekday', 'Weekend']
    my_pal = dict(zip((np.unique(df.index.day_name())), dayColors))

    for d in df.columns.values[3:5]:
        sns.boxplot(x=df.index.day_name(), y=d, data=df, palette=my_pal)  # weekday_name,month,day,hour
        for area, color in zip(seasons, np.unique(dayColors)): plt.scatter([], [], c=color, alpha=0.66, s=150,
                                                                           label=str(area))
        plt.legend(scatterpoints=1, frameon=False, labelspacing=1, title='Day Type')
        plt.ylim(0, 350)
        plt.show()


def cutAndCount(x): return pd.cut(x, AQScale, labels=categoryName).value_counts() / x.count()


def StackedBar(df):
    df = df.apply(cutAndCount).T
    df['grp'] = 4 * df['Hazardous'] + 2 * df['Very Unhealthy'] + df['Unhealthy']
    # df['grp'] = df['Good'] + df['Moderate'] + df['Unhealthy for Sensitive Groups']
    df = df.sort_values(by=['grp']).T
    df = df.drop('grp')

    datas = [go.Bar(x=df.columns.values, y=row,
                    marker_color=colorScale[categoryName.tolist().index(idx)],
                    name=idx) for idx, row in df.iterrows()]
    fig = go.Figure(data=datas)
    fig.update_layout(barmode='stack')
    fig.update_layout(legend_orientation="h")
    fig.show()


def allLagRange(x, df, lag, readings):
    def bestCorr(x, ss): return ss.apply(lambda y: y.corr(x)).argmax()

    ss = pd.concat([x.shift(shft) for shft in np.arange(lag + 1)], axis=1)
    readings = [df[reading[0]:reading[-1]] for reading in readings]
    newdf = pd.concat([reading.apply(bestCorr, ss=ss) for reading in readings], axis=1)
    newdf.columns = [str(reading.index.date[0]) + ' to ' + str(reading.index.date[-1]) for reading in readings]
    ser, ser.name = newdf.stack(), x.name
    return ser


def CrossCorrelation(df):
    df = df.rolling(center=True, window=4).mean()
    # df = df['2019-05':'2019-08']
    df = df['2020-01']
    print(df.isnull().sum())
    # MissingDataHeatmap(df)

    window, step, lag = 7, 1, 2
    readings = np.array(list(more_itertools.windowed(df.index[lag:-lag], n=window * 24, step=step * 24)))

    fileName, indices, freq = 'lagTimeMatrix_', ['Leader', 'Date', 'Follower'], '7D'
    lagTimeMatrix = df.apply(allLagRange, df=df, lag=lag, readings=readings).stack()
    lagTimeMatrix.index = lagTimeMatrix.index.rename(indices)
    lagTimeMatrix.to_csv(fileName + freq)

    lagTimeMatrix = pd.read_csv(fileName + freq, index_col=indices)
    lagTimeMatrix = lagTimeMatrix.unstack('Date')
    lagTimeMatrix.columns = lagTimeMatrix.columns.droplevel(0)
    print(lagTimeMatrix.shape)

    for key, df in lagTimeMatrix.iloc[:, :].items():
        if df.nunique() > 1: mapArrow(np.zeros(len(metaFrame)), df.unstack(), df.name)


def LatexFormatting(stats):
    latexData = stats.round(1).to_latex(col_space=3).replace("\\\n", "\\ \hline\n").replace('\\toprule',
                                                                                            '\\toprule\n\\hline')
    substring = latexData[
                latexData.index('\\begin{tabular}{') + len('\\begin{tabular}{') - 1:latexData.index('}\n') + 1]
    latexData = latexData.replace(substring, '|'.join(substring))

    for axisName in stats.columns: latexData = latexData.replace(axisName, f"\\textbf{{{axisName}}}")
    for axisName in stats.index: latexData = latexData.replace(axisName, f"\\textbf{{{axisName}}}")

    print(latexData)


if __name__ == '__main__':
    plt.close("all")
    sns.set()
    metaFrame, df = LoadMetadata(), LoadSeries()
    # LatexTableFormat(df.describe().T)
    # MissingDataHeatmap(df)
    # BoxPlotDistrict(df)
    MeteoAnalysis(df)

    # prof = ProfileReport(df['2017'].iloc[:,:10], minimal=False,title='Air Quality')
    # prof.to_file(output_file='Air Quality.html')

    # df = df.fillna(df.rolling(6, min_periods=1, ).mean()).round(3)

    # PLotlyTimeSeries(df[[cols]],missing[[cols]])
    # PLotlyTimeSeries(missing[[cols]],missing)

    # df = df.apply(FillMissingDataFromHours, args=[2])
    # df = df.apply(FillMissingDataFromDays, args=[3])
    # df = df.apply(FillMissingDataFromYears)

    # dfm = GetAllMeteoData()

    # SliderMapCommon(df['2017':'2020'], metaFrame, ['Y', '%Y %B'])
    # SliderMapCommon(df['2020'], metaFrame, ['D', '%Y %B %D'],True)

    # FrequencyClustering(df)
    exit()

    # for freq,data in df['2020'].resample('W').mean().iterrows():
    #     print(data)
    #     mapPlot(data,str(freq))

    # changePoints = np.hstack((([0],np.argwhere(np.diff((df.notnull().all(axis=1)).values)).squeeze(),[df.shape[0]-1]))).reshape(-1,2)
    # # print((changePoints[:,1]-changePoints[:,0])/(24*30))
    # cleanData = (changePoints[(changePoints[:,1]-changePoints[:,0])/(24*30)>1])
    # for cl in cleanData:print(df.index[cl[0]],df.index[cl[1]-1])

    # MissingDataHeatmap(df['2017'])
    # MissingDataHeatmap(df['2018'])
    # MissingDataHeatmap(df['2019'])
    # MissingDataHeatmap(df['2020'])

    # for cols in df.columns.values:
    #     days = 7
    #     ss = [df[cols].shift(shft, freq='H') for shft in np.delete(np.arange(-days,days+1), [-3+7,-2+7,-1+7,0+7,1+7,2+7,3+7])*24]
    #     df[cols] = df[cols].fillna((pd.concat(ss, axis=1).mean(axis=1)))

    # MissingBar(df)

    # corrArray = np.array([df['2017-12':'2017-12'].corr().values,
    #                       df['2018-03':'2018-03'].corr().values,
    #                       df['2018-06':'2018-06'].corr().values,
    #                       df['2018-09':'2018-09'].corr().values]).reshape((2, 2, df.shape[1], df.shape[1]))
    # CorrationSeasonal(corrArray)
    #
    # corrArray = np.array([df['2018-01':'2018-01'].corr().values,
    #                       df['2018-04':'2018-04'].corr().values,
    #                       df['2018-07':'2018-07'].corr().values,
    #                       df['2018-10':'2018-10'].corr().values]).reshape((2, 2, df.shape[1], df.shape[1]))
    # CorrationSeasonal(corrArray)
    #
    #
    # corrArray = np.array([df['2018-02':'2018-02'].corr().values,
    #                       df['2018-05':'2018-05'].corr().values,
    #                       df['2018-08':'2018-08'].corr().values,
    #                       df['2018-11':'2018-11'].corr().values]).reshape((2, 2, df.shape[1], df.shape[1]))
    # CorrationSeasonal(corrArray)
    #
    # corrArray = np.array([df['2017-12':'2018-02'].corr().values,
    #                       df['2018-03':'2018-05'].corr().values,
    #                       df['2018-06':'2018-08'].corr().values,
    #                       df['2018-09':'2018-11'].corr().values]).reshape((2, 2, df.shape[1], df.shape[1]))
    # CorrationSeasonal(corrArray)

    # corrArray = (np.array([[df['2017-'+str(month+1)].corr().values] for month in range(12)]).reshape((6, 2, df.shape[1], df.shape[1])))
    # CorrationSeasonal(corrArray,rows=6,title = '2017')
    #
    # corrArray = (np.array([[df['2018-'+str(month+1)].corr().values] for month in range(12)]).reshape((6, 2, df.shape[1], df.shape[1])))
    # CorrationSeasonal(corrArray,rows=6,title = '2018')
    #
    # corrArray = (np.array([[df['2019-'+str(month+1)].corr().values] for month in range(12)]).reshape((6, 2, df.shape[1], df.shape[1])))
    # CorrationSeasonal(corrArray,rows=6,title = '2019')

    [yearData, monthData, weekData, dayData] = [
        np.array([[timeDel, timeStamp, reading] for timeStamp, reading in
                  df.groupby(pd.Grouper(freq=timeDel)) if not reading.isnull().any().any()], dtype=object) for timeDel
        in
        (['Y', 'M', 'W', '3D'])]
    readings = dayData
    print(len(readings))

    totalEstimate = []
    for i, [timeDel, timeStamp, reading] in enumerate(readings[:]):
        print(i, timeStamp)
        # ratioMapPlotting(reading)
        # BoxPlotHour(reading)
        l1, l2 = CrossCorr(timeDel, timeStamp, reading, lagRange=2)
        totalEstimate.extend(l2)
        # BoxPlotSeason(reading)
        # BoxPlot(reading)
        # TriangularHeatmap(timeStamp,reading.astype('float64'))
    # WindGraphTeamEstimate(np.array(totalEstmare), ['Overall'])

    # for dis1 in metaFrame.index.values:
    #     for dis2 in metaFrame.index.values:
    #         lagRange = 3
    #         rs = [crosscorr(df[dis1], df[dis2], lag) for lag in range(-lagRange, 0)]
    #         offset = - int(np.floor(len(rs) / 2) - np.argmax(rs))
    #         if offset<0:
    #             print(offset)
    #             print(angleFromCoordinate(dis1, dis2))

    # SliderMapCommon(df, metaFrame, ['M', '%Y %B'])
    # PLotlyTimeSeries(df)
    # SliderMapCommon(df, metaFrame, ['M', '%Y %B'])
    # MeteoAnalysis(df)

    # ax = sns.boxplot(x=df.index.year, data=df.Dhaka.T)
    # pltSetUp("Month", "PM Reading", 'district', ylim=(0, 350))

    # df = df.resample('3D').max()
    # plt.figure(figsize=(9, 6))
    # ax = sns.boxplot(data=df.T, color="blue")
    # pltSetUpAx(ax, "Hour of Day", "PM Reading", 'district' + ' in ' + str('timeStamp'), ylim=(0, 500))
    # # pltSetUpAx(ax, "Hour of Day", "PM Reading", 'district' + ' in ' + str('timeStamp'))

    # # b = np.array([0, 4, 8, 12, 16, 20, 24])
    # b = np.array([pd.to_datetime(0, format='%H'), pd.to_datetime(12, format='%H')])
    # # l = ['Late Night', 'Early Morning', 'Morning', 'Noon', 'Eve', 'Night']
    # l = ['Day',  'Night']
    # df1['session'] = pd.cut(df.index, bins=b, labels=l)

    # MapPlotting(metaFrame, df[date].mean().values,vec=lagMatrix)
    # TriangularHeatmap(df)
    # SeasonalDecomposition(monthData[-1][2].iloc[:,3])

    # print(dayData[-1][1].shift(2))
    # print(dayData[-1][1].tshift(2))

    # print(df.info())
    # print(df.dtypes)
    # print(df.sample(1))
    # date_rng = pd.date_range(start='1/1/2018', end='1/08/2018', freq='H')

    # print([df1[2].mean().mean() for df1 in yearData])
    # popYear = [157977153, 159685424, 161376708, 163046161]
    # print(np.corrcoef([df1[2].mean().mean() for df1 in yearData][1:],popYear[1:]))
    # print(metaFrame[['Population','avgRead']].corr())

    # data.resample('S', fill_method='pad')  # forming a series of seconds

    # MissingDataHeatmap(df)
    # StackedBar(df)
    # BoxPlotDistrict(df, '')
    # BoxPlotMonthly(df['2017':'2019'])
    # for label, content in df.items():print(label,content)

    # print(df.to_markdown())
    # print(df.to_html())
    exit()
