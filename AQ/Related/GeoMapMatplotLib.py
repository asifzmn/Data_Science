import matplotlib.pyplot as plt
import pandas as pd
import shapefile as shp


def read_shapefile(sf):
    fields = [x[0] for x in sf.fields][1:]
    areaShapes = [s.points for s in sf.shapes()]
    return pd.DataFrame(columns=fields, data=sf.records()).assign(coords=areaShapes)


def plot_district_data(sf, metaFrame, data, vec, ratiodata, title, figsize, print_id=False, x_lim=None, y_lim=None):
    # df = read_shapefile(sf)
    # districtNames = (np.array(sf.records()))[:, 2]
    # districtReadings = [(np.where([districtNames == difflib.get_close_matches(d, districtNames)[0]])[-1][0]) for d in districts]

    plt.figure(figsize=figsize)
    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle(title, fontsize=25, va='top')

    for shape in sf.shapeRecords(): ax.plot([i[0] for i in shape.shape.points[:]],
                                            [i[1] for i in shape.shape.points[:]], 'k')
    #     if print_id != False:
    #         x0 = np.mean(x_lon)
    #         y0 = np.mean(y_lat)
    #         plt.text(x0 - len(districtNames[id]) / 75, y0, districtNames[id], fontsize=10)
    plt.show()


def MapPlotting(metaFrame=None, data=None, title='Map', figsize=(12, 18), vec=None, ratiodata=None):
    shp_path = '/media/az/Study/Air Analysis/Maps/bgd_admbnda_adm2_bbs_20180410/bgd_admbnda_adm2_bbs_20180410.shp'
    sf = shp.Reader(shp_path)

    if data is None: data = [45, 300, 299, 210, 125, 145, 175, 60, 125, 185, 10, 230][:3]
    if metaFrame is None: metaFrame = pd.DataFrame(columns=['Zone', 'Latitude', 'Longitude'],
                                                   data=[['Dhaka', 23.7298, 90.3854], ['Narsingdi', 22.9298, 91.1854],
                                                         ['Kushtia', 24.7298, 90.9854]])
    # if ratiodata is None: ratiodata = [[0.5, 0.25, 0.25],[0.25,0.25,0.2,0.2,0.1],[0.5, 0.17, 0.33]]

    print_id = True  # The shape id will be printed
    plot_district_data(sf, metaFrame, data, vec, ratiodata, title, figsize, print_id)


if __name__ == '__main__':
    MapPlotting()
    exit()
