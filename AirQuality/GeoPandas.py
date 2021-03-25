import json
import math
import xarray as xr
import pandas as pd
import geojsoncontour
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from geopy.distance import geodesic
from matplotlib import cm
from matplotlib.lines import Line2D
from plotly.offline import iplot
from AirQuality.DataPreparation import LoadMetadata, getCategoryInfo, LoadSeries, aq_directory
from scipy.interpolate import griddata
from numpy import linspace
import plotly.graph_objects as go
import plotly.express as px

from AirQuality.MeteoblueInfoAnalysis import getFactorData


def Distance(dis1, dis2):
    metaFrame = LoadMetadata()
    origin = metaFrame.loc[dis1]['Latitude'], metaFrame.loc[dis1]['Longitude']
    dest = metaFrame.loc[dis2]['Latitude'], metaFrame.loc[dis2]['Longitude']

    # x = geodesic(origin, dest).meters,geodesic(origin, dest).kilometers,geodesic(origin, dest).miles
    return geodesic(origin, dest).kilometers


def angleFromCoordinate(dis1, dis2):
    metaFrame = LoadMetadata()
    lat1, long1, lat2, long2 = metaFrame.loc[dis1]['Latitude'], metaFrame.loc[dis1]['Longitude'], metaFrame.loc[dis2][
        'Latitude'], metaFrame.loc[dis2]['Longitude']
    lat1, lat2, long1, long2 = math.radians(lat1), math.radians(lat2), math.radians(long1), math.radians(long2)

    dLon = (long2 - long1)
    y = math.sin(dLon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dLon)
    bearing = math.atan2(y, x)

    return (math.degrees(bearing) + 360) % 360


def draw_pie(ax, ratios, X=0, Y=0, size=1000):
    xy, start = [], 0
    colors = ["BURLYWOOD", "DARKBLUE"]

    for ratio in ratios:
        pieSegemnt = np.linspace(2 * math.pi * start, 2 * math.pi * (start + ratio), 30)
        x = [0] + np.cos(pieSegemnt).tolist()
        y = [0] + np.sin(pieSegemnt).tolist()
        print(x)
        print(y)
        xy.append(list(zip(x, y)))
        start += ratio

    for i, xyi in enumerate(xy):
        # print(X,Y)
        ax.scatter([X], [Y], marker=(xyi), s=size, facecolor=colors[i])


def Pie(ax, ratiodata):
    metaFrame = LoadMetadata()
    for i, txt in enumerate(metaFrame.index): draw_pie(ax, ratiodata[0], metaFrame.iloc[i]['Longitude'],
                                                       metaFrame.iloc[i]['Latitude'], size=800)

    lines = [Line2D([0], [0], color=c, linewidth=5, linestyle='-') for c in ["BURLYWOOD", "DARKBLUE"]]
    labels = ['Good or Moderate', 'Unhealthy or Hazardous']
    plt.legend(lines, labels, loc='lower left', prop={'size': 15})


def Arrows(ax, vec):
    colorGen = cm.get_cmap('Purples', 256)
    # newcolors = colorGen(np.linspace(0, 256, 7))
    # pal = colorGen(np.linspace(0, 1, 15))[4:11][::-1]
    pal = colorGen(np.linspace(0, 1, 5))[1:5]
    # pal = []

    metaFrame = LoadMetadata()
    for x in range(len(vec)):
        for y in range(len(vec)):
            # if vec[x][y] == 0.0: continue
            if vec.iloc[x][y] <= 0: continue
            # c = 'b' if vec[x][y] > 0 else 'r'
            ax.arrow(metaFrame.iloc[x]['Longitude'], metaFrame.iloc[x]['Latitude'],
                     metaFrame.iloc[y]['Longitude'] - metaFrame.iloc[x]['Longitude'],
                     metaFrame.iloc[y]['Latitude'] - metaFrame.iloc[x]['Latitude'],  # width=.01 * vec.iloc[x][y],
                     # metaFrame.iloc[y]['Latitude'] - metaFrame.iloc[x]['Latitude'], width=.001 * 3,
                     head_width=.075, head_length=0.1, length_includes_head=True, zorder=1,
                     color=pal[int(vec.iloc[x][y])],
                     # head_width=.045, head_length=0.1, length_includes_head=True, zorder=1, color=pal[-1],
                     ls='-')
            # head_width=.033, head_length=0.081, length_includes_head=True, zorder=0, color = 'grey',ls = '-')


def setMap(x=2):
    plt.rcParams['figure.figsize'] = (8 * x, 10 * x)
    # df_admin = gpd.read_file('/media/az/Study/Air Analysis/Maps/districts.geojson')
    df_admin = gpd.read_file(aq_directory + 'Maps/districts.geojson')
    return df_admin.plot(color='#E5E4E2', edgecolor='#837E7C')


def MapScatter(ax, data=None):
    colorScale, categoryName, AQScale = getCategoryInfo()

    # data = np.arange(15, 15 + 22 * 9, 9)
    # color_ton = [colorScale[val] for val in np.digitize(data, AQScale[1:-1])]

    # colorScale = ['#C38EC7','#E42217','#FFD801','#5EFB6E','#5CB3FF','#34282C']
    # color_ton = [colorScale[val] for val in np.digitize(data, AQScale[1:-1])]

    # ax.scatter(x=data.Longitude.astype('float64'), y=data.Latitude.astype('float64'), zorder=1, alpha=1,
    #            c=data.color, s=150, marker='H', edgecolor='#3D3C3A', linewidth=1)

    for idx, row in data.iterrows():
        marker_size, marker_color, = 150, '#566D7E'
        if idx in ['Sirajganj', 'Jessore', 'Dhaka']: marker_size, marker_color = 450,'#5b567e'
        ax.scatter(x=row.Longitude, y=row.Latitude, zorder=1, alpha=1,
                   c=marker_color, s=marker_size, marker=row.symbol, edgecolor='#3D3C3A', linewidth=1)


def MapLegend(ax, legendData):
    # lines = [Line2D([0], [0], color=c, linewidth=5, linestyle='-') for c in legendData[1].color]
    lines = [Line2D([], [], color='#566D7E', marker=s, linestyle='None', markersize=15) for s in legendData[1].symbol]
    ax.legend(lines, legendData[1].category, loc='lower left', prop={'size': 15}, title=legendData[0])


def MapAnnotate(ax, data):
    for idx, row in (data.iterrows()): ax.annotate(idx, (row.Longitude - len(idx) * .015, row.Latitude - .1),
                                                   fontsize=15)


def mapPlot(data, legendData, save=None):
    ax = setMap()
    MapScatter(ax, data)
    MapAnnotate(ax, data)
    MapLegend(ax, legendData)
    plt.tight_layout()
    plt.show()
    if save is not None:
        plt.savefig(save + '.png', dpi=300)
        plt.clf()


def mapArrow(data, mat, times, save=None):
    ax = setMap()
    MapScatter(ax, data)
    MapAnnotate(ax, data)
    Arrows(ax, mat)
    # plt.title('From '+str(times[0])+' to '+str(times[-1]))
    plt.title(times)
    plt.tight_layout()
    plt.show()

    if save is not None:
        plt.savefig(save + '.png', dpi=300)
        plt.clf()


def heatmapgeoJson(metaFrame, title):
    with open('/home/az/Desktop/bdBounds.geojson') as file: bdBounds = json.load(file)

    rounding_num, correction_coeff, segments, regions = 0.015, 0.5, 500, 15
    metaFrame["Longitude"] = np.round(metaFrame["Longitude"] / rounding_num) * rounding_num
    metaFrame["Latitude"] = np.round(metaFrame["Latitude"] / (rounding_num * correction_coeff)) * (
            rounding_num * correction_coeff)

    z, y, x = metaFrame['Value'], metaFrame['Latitude'], metaFrame['Longitude']
    xi, yi = linspace(x.min(), x.max(), segments), linspace(y.min(), y.max(), segments)
    zi = griddata((x, y), z, (xi[None, :], yi[:, None]))

    step_size = math.ceil((np.nanmax(zi) - np.nanmin(zi)) / regions)
    cs = plt.contourf(xi, yi, zi, range(int(np.nanmin(zi)), int(np.nanmax(zi)) + step_size + 1, step_size))

    price_geojson = eval(geojsoncontour.contourf_to_geojson(contourf=cs, ndigits=10, ))

    price_geojson["features"] = [price_geojson["features"][i] for i in range(len(price_geojson["features"])) if
                                 len(price_geojson["features"][i]['geometry']['coordinates']) > 0]

    arr_temp = np.ones([len(price_geojson["features"]), 2])

    for i in range(len(price_geojson["features"])):
        price_geojson["features"][i]['properties']["id"] = i
        arr_temp[i] = i, float(price_geojson["features"][i]["properties"]["title"].split('-')[0])

    df_contour = pd.DataFrame(arr_temp, columns=["Id", "Value"])
    center_coors = 23.6850, 90.3563

    district = (go.Scattermapbox(
        mode="markers", showlegend=False,
        lon=metaFrame['Longitude'], lat=metaFrame['Latitude'], text=metaFrame.index,
        marker=go.scattermapbox.Marker(size=6, color='#43BFC7')
    ))

    districtBorder = (go.Scattermapbox(
        mode="markers", showlegend=False,
        lon=metaFrame['Longitude'], lat=metaFrame['Latitude'],
        marker=go.scattermapbox.Marker(size=9, color='#504A4B')
    ))

    df_bd = pd.DataFrame({'Value': [0], 'Id': [19]})
    tracebd = go.Choroplethmapbox(
        geojson=bdBounds, featureidkey="properties.ID_0",
        locations=df_bd.Id, z=df_bd.Value,
        marker=dict(opacity=0.15), colorscale='greys', showscale=False
    )

    trace = go.Choroplethmapbox(
        geojson=price_geojson, z=df_contour.Value,
        locations=df_contour.Id, featureidkey="properties.id",
        marker=dict(opacity=0.45), marker_line_width=0,
        zmin=0, zmax=250,
        # zmin=25, zmax=35,
        # colorscale=[(0, '#46d246'), (0.05, '#46d246'), (0.05, '#ffff00'), (0.14, '#ffff00'),
        #             (0.14, '#ffa500'), (0.22, '#ffa500'), (0.22, '#ff0000'), (0.6, '#ff0000'),
        #             (.6, '#800080'), (1, '#800080')],  # Discreet

        colorscale=[(0, '#46d246'), (0.045, '#1b701b'), (0.055, '#ffff00'), (0.135, '#7f7f00'),
                    (0.145, '#ffa500'), (0.22, '#7f5200'), (0.25, '#ff0000'), (0.55, '#7f0000'),
                    (0.65, '#800080'), (1, '#400040')],  # Discreet with continuous in group
    )

    layout = go.Layout(
        title=title, title_x=0.4,
        width=750,
        margin=dict(t=80, b=0, l=0, r=0),
        font=dict(color='dark grey', size=18),
        mapbox=dict(
            center=dict(lat=center_coors[0], lon=center_coors[1]),
            zoom=6.5,
            style="carto-positron"
        )
    )

    figure = dict(data=[tracebd, trace, districtBorder, district], layout=layout)
    iplot(figure)


if __name__ == '__main__':
    # meteoData = xr.open_dataset('meteoData.nc')['meteo']

    metaFrame, df = LoadMetadata(), LoadSeries()['2018-01':'2018-12'].resample('W').mean()
    # metaFrame, df = LoadMetadata(), getFactorData(meteoData, 'Temperature [2 m]')

    for i, row in df.sample(5).iterrows():
        if row.isnull().any(): continue
        metadata = metaFrame.assign(Value=row)
        # title = (str(i.date()-pd.Timedelta(days=7))+" "+str(i.date()))
        title = (str(i.date()))
        heatmapgeoJson(metadata, title)
    exit()

    ax = setMap()
    # MapScatter(ax)
    # Pie(ax,[[.5,.5]])
    # Arrows(ax,mat)
    # MapAnnotate(ax)
    plt.show()
    exit()
