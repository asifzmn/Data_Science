import plotly.graph_objects as go

from AirQuality.DataPreparation import LoadData

if __name__ == '__main__':
    # with open('bangladesh.geojson') as file : data = json.load(file)
    #
    # # print(len(data['features']))
    # for x in (data['features'][:100]):print(x['properties'])
    #
    # z = [x['properties']['ID_3']  for x in (data['features'])]
    # print((np.unique(z)))
    # exit()
    #
    # with open('/home/az/Desktop/geojson-counties-fips.json') as response:counties = json.load(response)
    # print(len(counties['features']))
    # for x in (counties['features'][:1]):print(x['properties'])
    #
    #
    # df = pd.read_csv("/home/az/Desktop/fips-unemp-16.csv",
    #                    dtype={"fips": str})
    # # print((df))
    # exit()
    #
    # fig = go.Figure(go.Choroplethmapbox(geojson=data, locations=df.fips, z=df.unemp,
    #                                     colorscale="Viridis", zmin=0, zmax=12,
    #                                     marker_opacity=0.5, marker_line_width=0))
    #
    # # fig = go.Figure(go.Choroplethmapbox(geojson=counties, locations=df.fips, z=df.unemp,
    # #                                     colorscale="Viridis", zmin=0, zmax=12,
    # #                                     marker_opacity=0.5, marker_line_width=0))
    #
    # fig.update_layout(mapbox_style="carto-positron",
    #                   mapbox_zoom=3, mapbox_center = {"lat": 37.0902, "lon": -95.7129})
    # fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    # fig.show()
    #

    dataVector, metaFrame = LoadData(name='reading1.pickle')
    # df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2011_february_us_airport_traffic.csv')
    # df['text'] = df['airport'] + '' + df['city'] + ', ' + df['state'] + '' + 'Arrivals: ' + df['cnt'].astype(str)

    fig = go.Figure(data=go.Scattergeo(
        lon=metaFrame['Longitude'],
        lat=metaFrame['Latitude'],
        text=metaFrame['Zone'],
        mode='markers',
        marker_color=metaFrame['Population'],
    ))

    fig.update_layout(
        title='Most trafficked US airports<br>(Hover for airport names)',
        geo_scope='asia',
    )
    # fig.update_layout(mapbox_style="carto-positron",
    #                   mapbox_zoom=3, mapbox_center={"lat": 37.0902, "lon": -95.7129})
    fig.show()

    # with open("bangladesh.geojson") as json_file:topoJSON = json.loads(json_file.read())
    # # print(topoJSON.keys())
    # # print(topoJSON['type'])
    # # print(len(topoJSON['features']))
    # # print((topoJSON['features'][0].keys()))
    # # print((topoJSON['features'][0]['geometry']))
    # # print((topoJSON['features'][0]['properties']))
    # # print((topoJSON['features'][2]))
    #
    # pts = []  # list of points defining boundaries of polygons
    # for feature in topoJSON['features'][:10]:
    #     print(feature['geometry']['type'])
    #     if feature['geometry']['type'] == 'Polygon':
    #         pts.extend(feature['geometry']['coordinates'][0])
    #         pts.append([None, None])  # mark the end of a polygon
    #
    #     elif feature['geometry']['type'] == 'MultiPolygon':
    #         for polyg in feature['geometry']['coordinates']:
    #             pts.extend(polyg[0])
    #             pts.append([None, None])  # end of polygon
    #
    # X, Y = zip(*pts)
    #
    # data = [dict(type='scatter',
    #              x=X,
    #              y=Y,
    #              mode='lines',
    #              line=dict(width=0.5, color='blue'),
    #              )]
    # axis_style = dict(showline=False,
    #                   mirror=False,
    #                   showgrid=False,
    #                   zeroline=False,
    #                   ticks='',
    #                   showticklabels=False)
    # layout = dict(title='France regions',
    #               width=700, height=700,
    #               autosize=False,
    #               xaxis=axis_style,
    #               yaxis=axis_style,
    #               hovermode='closest')
    # fig = dict(data=data, layout=layout)
    # # fig.show()
    # py.iplot(fig, filename='France-map2d')

    # mapbox_access_token = open(".mapbox_token").read()
    #
    # fig = go.Figure(go.Scattermapbox(
    #     lat=['45.5017'],
    #     lon=['-73.5673'],
    #     mode='markers',
    #     marker=go.scattermapbox.Marker(
    #         size=14
    #     ),
    #     text=['Montreal'],
    # ))
    #
    # fig.update_layout(
    #     hovermode='closest',
    #     mapbox=dict(
    #         accesstoken=mapbox_access_token,
    #         bearing=0,
    #         center=go.layout.mapbox.Center(
    #             lat=45,
    #             lon=-73
    #         ),
    #         pitch=0,
    #         zoom=5
    #     )
    # )
    #
    # fig.show()
