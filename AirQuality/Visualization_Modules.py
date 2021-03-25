import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from statsmodels.tsa.seasonal import seasonal_decompose
from AirQuality.DataPreparation import *


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


def PairDistributionSummary(timeseries, samplingHours=1):
    # df = df[: str(max(df.index.date) + timedelta(days=-1))].resample(str(samplingHours) + 'H').mean()
    timeseries = timeseries.resample(str(samplingHours) + 'H').mean()
    timeseries['daytime'] = np.tile(
        np.hstack((np.repeat('Day', 12 // samplingHours), np.repeat('Night', 12 // samplingHours))),
        ((timeseries.shape[0] * samplingHours) // 24))

    timeseries = timeseries.sort_values(by=['daytime'], ascending=False)
    g = sns.pairplot(timeseries, hue='daytime', palette=["#2B3856", "#FFF380"], plot_kws={"s": 9})

    for ax in plt.gcf().axes: ax.set_xlabel(ax.get_xlabel(), fontsize=30)
    for ax in plt.gcf().axes: ax.set_ylabel(ax.get_ylabel(), fontsize=30)

    g.fig.get_children()[-1].set_bbox_to_anchor((1.1, 0.5, 0, 0))
    plt.show()


def BoxPlotYear(df):
    for district in df:
        ax = sns.boxplot(x=df.index.year, y=district, data=df, color="#00AAFF")  # weekday_name,month,day,hour
        pltSetUpAx(ax, "Year", "PM Reading", 'Yearly average reading in ' + district, ylim=(0, 200))


def BoxPlotSeason(timeseries):
    month_colors = ["#8fcadd"] * 2 + ["#46d246"] * 3 + ["#ff0000"] * 3 + ["#ffa500"] * 3 + ["#8fcadd"]
    seasons = ['Spring', 'Winter', 'Summer', 'Autumn']
    seasonPalette = dict(zip((np.unique(timeseries.index.month)), month_colors))

    for zone in timeseries:
        plt.figure(figsize=(8, 8))
        ax = sns.boxplot(x=timeseries.index.month, y=zone, data=timeseries,
                         palette=seasonPalette)  # weekday_name,month,day,hour
        for seasonName, color in zip(seasons, np.unique(month_colors)): plt.scatter([], [], c=color, alpha=0.66, s=150,
                                                                                    label=str(seasonName))
        plt.legend(scatterpoints=1, frameon=False, labelspacing=.5, title='Season')
        pltSetUpAx(ax, xlabel="Month", ylabel="PM Reading", title='Seasonality of months in ' + zone, ylim=(0, 300))


def BoxPlotHour(timeseries):
    color_map = LinearSegmentedColormap.from_list('DayNight',
                                                  ["#075077", "#075077", "#6f5a66", "#6f5a66",
                                                   "#eae466", "#eae466", "#eae466", "#eae466",
                                                   "#6f5a66", "#6f5a66", "#075077", "#075077"], N=24)
    hourPalette = dict(zip((np.unique(timeseries.index.hour)), color_map(np.arange(24))))
    for zone in timeseries.columns.values[:]:
        plt.figure(figsize=(8, 8))
        ax = sns.boxplot(x=timeseries.index.hour, y=zone, data=timeseries,
                         palette=hourPalette)  # weekday_name,month,day,hour
        pltSetUpAx(ax, xlabel="Hour of day", ylabel="PM Reading", title='HourSeasonality' + zone, ylim=(0, 300))


def pltSetUp(xlabel=None, ylabel=None, title=None, xlim=None, ylim=None, save=None):
    if not xlabel is None: plt.xlabel(xlabel)
    if not ylabel is None: plt.ylabel(ylabel)
    if not title is None: plt.title(title)
    if not xlim is None: plt.xlim(xlim[0], xlim[1])
    if not ylim is None: plt.ylim(ylim[0], ylim[1])

    if save is None:
        plt.show()
    else:
        plt.savefig(f"{save} {title}.png", dpi=300)
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
        plt.savefig(f"{save} {title}.png", dpi=300)
        plt.clf()
        print(f"{save} {title}.png")


def SimpleTimeseries(df):
    resampled_df = df.resample('M')
    # resampled_df = df.resample('D')
    aggregated_value = pd.concat(
        [sampledSeries.stack().apply(['min', 'mean', 'median', 'max']) for _, sampledSeries in resampled_df], axis=1).T
    aggregated_value.index = [time for time, _ in resampled_df]
    aggregated_value = aggregated_value
    # aggregated_value = aggregated_value.iloc[:-1]
    print(aggregated_value.to_string())
    aggregated_value.to_csv('aggregated_value.csv')

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=aggregated_value.index.tolist() + aggregated_value.index[::-1].tolist(),
        y=aggregated_value['max'].tolist() + aggregated_value['min'].tolist()[::-1],
        fill='toself', showlegend=False,
        fillcolor='rgba(127, 127, 127,0.3)',
        line_color='rgba(255,255,255,0)',
    ))

    fig.add_trace(go.Scatter(
        x=aggregated_value.index.tolist(), y=aggregated_value['mean'].tolist(),
        line_color='rgb(0, 38, 255)', line_width=5, name='Mean',
    ))

    fig.add_trace(go.Scatter(
        x=aggregated_value.index.tolist(), y=aggregated_value['median'].tolist(),
        line_color='rgb(255, 216, 1)', line_width=5, name='Median',
    ))

    # # annotate_points = ['2017-01-01', '2018-01-14', '2019-01-13', '2017-07-02', '2018-07-15', '2019-07-14']
    # annotate_points = ['2017-01-31', '2018-01-31', '2019-01-31', '2017-07-31', '2018-07-31', '2019-07-31']
    # # annotate_points = ['2017-01-31', '2018-01-31', '2019-01-31', '2017-07-31', '2018-07-31', '2019-07-31','2017-12-31', '2018-12-31', '2019-12-31']
    # for annotate_point in annotate_points:
    #     print(aggregated_value.loc[annotate_point, 'median'])
    #     fig.add_annotation(x=annotate_point, y=aggregated_value.loc[annotate_point, 'median'],
    #                        text=aggregated_value.loc[annotate_point, 'median'])

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#E5E4E2')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E5E4E2')

    fig.update_traces(mode='lines')
    fig.update_layout(
        yaxis_title="PM2.5 Concentration",
        xaxis_title="Time",
        font=dict(size=27, ),
        legend_orientation="h",
        template='plotly_white'
    )
    fig.show()


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


def ViolinPLot(df):
    fig = go.Figure()
    # df = df.resample('M').mean()
    years = ['2017', '2018', '2019']

    for year in years:
        fig.add_trace(go.Violin(y=df[year].stack(),
                                name=year, box_visible=True,
                                meanline_visible=True, line_color='#566D7E',

                                )
                      )
    fig.update_layout(font=dict(size=21), width=900,
                      title="Air quality of Bangladesh over years",
                      yaxis_title="PM2.5 Concentration",
                      xaxis_title="Year",
                      )
    fig.show()


def ColorTable():
    colorScale, categoryName, AQScale = getCategoryInfo()
    range = [str(a) + " - " + str(b) for a, b in zip(AQScale, AQScale[1:])]
    data = {'Category': categoryName, 'Color': colorScale, 'range': range}
    df = pd.DataFrame(data)

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=["<b>Category</b>", "<b>Concentration Range (&#956;gm<sup>-3</sup>) </b>"],
            line_color='grey', fill_color='silver',
            align='center', font=dict(color='black', size=15)
        ),
        cells=dict(
            values=[df.Category, df.range],
            line_color='grey', fill_color=[df.Color],
            align='center', font=dict(color='black', size=12)
        ))
    ])

    fig.update_layout(width=666)

    fig.show()


if __name__ == '__main__':
    plt.close("all")
    # sns.set()
    # # sns.set_style("whitegrid")
    # metaFrame, df = LoadMetadata(), LoadSeries()['2017':'2019']
