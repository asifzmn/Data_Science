import pandas as pd
import plotly.graph_objects as go


def processTS(path): return pd.read_csv(path, index_col='Time', low_memory=False, parse_dates=[0])


def createAndSaveTimeSeries(df, areaName):
    def dateSeries(x, date_rng): return x.set_index(month, drop=True).reindex(date_rng)[value]

    acc, month, value = df.columns.values
    date_rng = pd.date_range(start=min(df[month]), end=max(df[month]), freq='MS')
    df = df.groupby(acc).apply(dateSeries, date_rng=date_rng).T
    df.index.name = 'Time'
    df.to_csv(areaName)


def PlotUsageTimeseries(histo, title):
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
