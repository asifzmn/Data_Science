import pandas as pd
from AirQuality.DataPreparation import LoadSeries, LoadMetadata
from geopy.distance import geodesic
import plotly.express as px
import numpy as np


def MapPlotIndustries(df):
    px.set_mapbox_access_token(
        'pk.eyJ1IjoiaG9vbmtlbmc5MyIsImEiOiJjam43cGhpNng2ZmpxM3JxY3Z4ODl2NWo3In0.SGRvJlToMtgRxw9ZWzPFrA')

    fig = px.scatter_mapbox(df, lat="lat", lon="lon", color='queryZone', hover_data=['category'])
    fig.show()


if __name__ == '__main__':
    print((92.33122375401729 - 88.4313463096257) / 0.0633803)
    print((25.292835653941832 - 22.15395039661749) / 0.0633803)

    print((92.33122375401729 - 88.4313463096257) / 0.1398059)
    print((25.292835653941832 - 22.15395039661749) / 0.1398059)

    L = 90.33560946128588
    R = 90.46136001989862
    U = 23.90779375397438
    D = 23.70748592269221

    print((R - L))
    print((R - L) / 0.0633803)
    # print((U-D)/0.0633803)

    print(geodesic((23.9303329, 90.7705742), (23.9303329, 90.7071939)))
    exit()
    # df = pd.read_csv('manufacturing_Dhaka.csv')
    names = ['Faridpur', 'Gazipur', 'Dhaka', 'Kishorganj', 'Madaripur', 'Narayanganj', 'Narsingdi', 'Mymensingh',
             'Barisal', 'Sirajganj', 'Tangail'][:]
    ZoneSel = names[-9:]
    metaFrame, df = LoadMetadata().loc[ZoneSel], LoadSeries()['2019']
    # print(df)
    # print(metaFrame)

    # manufacturingInfo = {name:pd.read_csv(f'manufacturing_{name}.csv').iloc[:] for name in names}
    manufacturingInfoAll = pd.concat(
        [pd.read_csv(f'manufacturing_{name}.csv', index_col='Unnamed: 0') for name in names])
    manufacturingInfoAll = manufacturingInfoAll.drop_duplicates().fillna('')[['lat', 'lon', 'category', 'queryZone']]
    # print(manufacturingInfoAll.category.value_counts().to_string())
    manufacturingInfoAll = manufacturingInfoAll[
        manufacturingInfoAll.category.str.contains('|'.join(['manufacturer', 'mill', 'power']), case=False)]
    # manufacturingInfoAll = manufacturingInfoAll[manufacturingInfoAll.category.str.contains('manufacturer',case=False)|
    #                                             manufacturingInfoAll.category.str.contains('mill',case=False)]
    # print(manufacturingInfoAll.to_string())

    MapPlotIndustries(manufacturingInfoAll)
    print(metaFrame)
    exit()

    # metaFrame['distance'] = metaFrame.apply(lambda x: manufacturingInfo[x.name].apply(lambda y: geodesic((x.Latitude,x.Longitude), (y.lat,y.lon)).kilometers,axis = 1).mean() , axis = 1)
    metaFrame['reading'] = df[ZoneSel].median()
    # print(manufacturingInfoAll)

    distanceMatrix = metaFrame.apply(
        lambda x: manufacturingInfoAll.apply(lambda y: geodesic((x.Latitude, x.Longitude), (y.lat, y.lon)).kilometers,
                                             axis=1), axis=1)
    metaFrame['distance'] = (distanceMatrix.apply(lambda x: x.sort_values()[:15].mean(), axis=1))

    metaFrame = metaFrame.sort_values('distance')
    print(metaFrame)
    fig = px.scatter(metaFrame, x="reading", y="distance", hover_data=[metaFrame.index])
    fig.show()

    # print(manufacturingInfo.columns)
    # print(manufacturingInfoAll.category.value_counts().to_string())
