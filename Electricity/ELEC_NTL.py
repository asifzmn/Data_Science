import pandas as pd
import json
import plotly.express as px


def Choropleth(df):
    with open('/media/az/Study/Datasets/Bangladesh/nocs.geojson') as json_file:
        counties = json.load(json_file)
    a = counties['features'][0]['geometry']['coordinates']
    print(a)

    fig = px.choropleth_mapbox(df, geojson=counties, locations='index', color=df.columns[1],
                               color_continuous_scale="Viridis", featureidkey="properties.NOCS",
                               range_color=(-1, 1),
                               mapbox_style="carto-positron",
                               zoom=7, center={"lat": 23.8103, "lon": 90.4125},
                               opacity=0.5,
                               )

    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.show()

    fig = px.choropleth(df, geojson=counties, color="sumCorr",
                        locations='index', featureidkey="properties.NOCS",
                        projection="mercator"
                        )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.show()


if __name__ == '__main__':

    dfNTL = pd.read_csv('/home/asif/Work/Projects/NTL Docs/nocsNTL.csv', index_col='NOCS').drop(['Started', 'type'], axis=1).astype(
        'float64').T
    # dfElec = pd.read_csv('Data Directory/NOCSELEC2.csv', index_col='MONTH')
    # dfElec = pd.read_csv('/media/az/Study/Datasets/Electricity/Data Directory/NOCSELEC3.csv', index_col='Unnamed: 0', parse_dates=[0])
    # dfElec.rename(
    #     columns={'Azimpur': 'AZIMPUR', 'SHAMOLY': 'SHYAMOLI', 'SHERE-B-NAGAR': 'SHERE B.NAGAR', 'Lalbag': 'LALBAG'},
    #     inplace=True)
    # dfNTL = dfNTL.T[dfElec.columns].astype('float64')


    # dfNTL = dfNTL[dfNTL.columns.drop(list(dfNTL.filter(regex='variance')))]
    # dfNTL = dfNTL.T.loc[['2018_sum', '2019_sum']][dfElec.columns].astype('float64')
    # dfNTLsum,dfNTLmean = dfNTL.iloc[1::3],dfNTL.iloc[2::3]

    dfNTLsum, dfNTLmean = dfNTL.iloc[19::3], dfNTL.iloc[20::3]
    # dfNTLsum.index,dfNTLmean.index = pd.to_datetime(dfNTLsum.index.str[:4]),pd.to_datetime(dfNTLmean.index.str[:4])
    dfNTLsum.index, dfNTLmean.index = pd.to_datetime(dfNTLsum.index.str[:10]), pd.to_datetime(dfNTLmean.index.str[:10])
    print(dfNTLsum)
    print(dfNTLmean)
    # print(dfElec)

    dfNTLsum.to_csv('NOCS_NTL_sum_2018.csv')
    dfNTLmean.to_csv('NOCS_NTL_mean_2018.csv')

    # print(dfNTLsum,dfNTLmean)

    # PLotlyTimeSeries(dfElec)
    # PLotlyTimeSeries(dfNTLsum)
    # PLotlyTimeSeries(dfNTLmean)
    exit()

    sumCorr, meanCorr = dfElec.corrwith(dfNTLsum, axis=0), dfElec.corrwith(dfNTLmean, axis=0)
    sumCorr, meanCorr = sumCorr.round(3), meanCorr.round(3)
    sumCorr.name, meanCorr.name = 'Correlation with sum NTL', 'Correlation with mean NTL'

    # print(sumCorr)
    # print(meanCorr)
    # print(sumCorr.to_latex(col_space=3).replace("\\\n", "\\ \hline\n"))
    # print(meanCorr.to_latex(col_space=3).replace("\\\n", "\\ \hline\n"))
    # print(pd.concat([meanCorr,sumCorr],axis=1).to_html())


    # Choropleth(sumCorr.reset_index())

    exit()
    # print(dfNTL)
    # print(dfElec)

    for x in range(3):
        dfNTLpart = dfNTL.iloc[x::3]
        print(dfNTLpart.index)

        # print(dfElec.columns)
        # print(dfNTL.columns)
        # dfNTLpart = dfNTLpart.mean(axis=0)

        for i in range(6):
            s1 = dfElec.iloc[i]
            s2 = dfNTLpart.iloc[i]
            # s2 = dfNTL

            print(s1.corr(s2, method='pearson'))

        print()

    dfNTL = dfNTL.iloc[2::3]

    print(dfElec.T.to_string())
    print(dfNTL.T.to_string())

    for d in dfNTL.columns:
        # print(dfNTL[d])
        dfNTL[d].index = dfElec.index
        print(d, dfNTL[d].corr(dfElec[d], method='pearson'))

    exit()
