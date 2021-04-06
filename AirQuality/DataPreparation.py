from datetime import datetime, timedelta
from os import listdir
from os.path import join, isfile
from urllib.request import urlopen
import pandas as pd
import numpy as np
import pickle as pk
import pandas as pd

[Year, Month, Day, UTC_Hour, PM25, PM10_mask, Retrospective] = range(7)
discard = 'Azimpur,Bhola,Patiya,Laksham,Netrakona,Madaripur,Ishurdi,Pabna,Tungipara,Ramganj,Raipur,Palang,Sherpur,Nagarpur,Sarishabari,Shahzadpur,Pirojpur,Maulavi_Bazar,Habiganj,Bhairab_Bazar,Sandwip,Satkania,Rangpur,Khagrachhari,Lakshmipur,Jamalpur,Saidpur,Chittagong,Lalmanirhat,Thakurgaon,Sylhet,Dinajpur'.split(
    ',')

# aq_directory = '/home/asif/Work/Air Analysis/AQ Dataset/'
aq_directory = '/media/az/Study/Air Analysis/AQ Dataset/'
berkely_earth_data = aq_directory + 'Berkely Earth Data/'
berkely_earth_data_prepared = berkely_earth_data + 'prepared/'
meteoblue_data_path = aq_directory + 'Meteoblue Scrapped Data/'


def getCommonID(id=0): return ['all', 'SouthAsianCountries', 'allbd'][id]


def getZones(): return pd.read_csv(
    berkely_earth_data + 'zones/' + getCommonID() + '.csv')


def getCategoryInfo():
    colorScale = np.array(['#46d246', '#ffff00', '#ffa500', '#ff0000', '#800080', '#6a2e20'])
    lcolorScale = np.array(['#a2e8a2', '#ffff7f', '#ffd27f', '#ff7f7f', '#ff40ff', '#d38370'])
    dcolorScale = np.array(['#1b701b', '#7f7f00', '#7f5200', '#7f0000', '#400040', '#351710'])
    categoryName = np.array(
        ['Good', 'Moderate', 'Unhealthy for Sensitive Groups', 'Unhealthy', 'Very Unhealthy', 'Hazardous'])
    AQScale = np.array([0, 12.1, 35.5, 55.5, 150.5, 250.5, 500])
    return colorScale, categoryName, AQScale


def TimetoString(time, timeScale): return (np.array([(str(s.strftime(timeScale[2])).split('-')) for s in time])).astype(
    'int').astype('str')


def getCommonTimes(allDistrictData, timeKey):
    print(timeKey)
    timeScale = {'year': (Year, Year, '%Y'), 'month': (Month, Month, '%m'), 'day': (Day, Day, '%d'),
                 'hour': (UTC_Hour, UTC_Hour, '%H'), 'yearmonth': (Year, Month, '%Y-%m'),
                 'monthday': (Month, Day, '%m-%d'), 'yearmonthday': (Year, Day, '%Y-%m-%d'),
                 'yearmonthdayhour': (Year, UTC_Hour, '%Y-%m-%d-%H')}[timeKey]
    datetimes = np.array(
        [[datetime.strptime('-'.join(map(str, x)), timeScale[2]) for x in d[:, timeScale[0]:timeScale[1] + 1]] for d in
         allDistrictData], dtype='object')

    # for a,d in zip(allDistrictMetaData[:, 0],datetimes):print(a,len(np.unique(TimetoString(d,timeScale))))
    sortedCommonTime = np.array(sorted(set.intersection(*map(set, [[x for x in d] for d in datetimes]))))

    if timeKey == 'yearmonthdayhour' or timeKey == 'yearmonthday':
        minutes = 60 if timeKey == 'yearmonthdayhour' else 24 * 60
        startTime, endTime = np.min([np.min(d) for d in (datetimes)]), np.max([np.max(d) for d in (datetimes)])
        # startTime, endTime = np.min(sortedCommonTime), np.max(sortedCommonTime)
        continuousTime = np.array([(startTime + timedelta(seconds=i)) for i in
                                   range(0, int((endTime - startTime).total_seconds()) + 60 * minutes,
                                         int(timedelta(minutes=minutes).total_seconds()))])
        # continuousTimeStr = TimetoString(continuousTime, timeScale)
        continuousTimeData = (np.full((len(allDistrictData), len(continuousTime)), None))
        for i, districtData in enumerate(allDistrictData):
            cur = 0
            for j, time in enumerate(continuousTime):
                if cur < len(districtData) and datetimes[i][cur] < time: cur += 1

                commonHourReading = []
                while cur < len(districtData) and datetimes[i][cur] == time:
                    commonHourReading.append(districtData[cur].astype('float64')[PM25])
                    cur += 1
                if not len(commonHourReading) == 0: continuousTimeData[i][j] = np.mean(np.array(commonHourReading))
        return continuousTime, np.array(continuousTimeData)
        # return dataInterpolation((continuousTimeStr, np.array(continuousTimeData)))

    return sortedCommonTime, (np.array([[(np.mean((districtData[:, PM25].astype('float64')[
        np.transpose(np.where(np.all(districtData[:, timeScale[0]:timeScale[1] + 1] == ct, axis=1)))])[:, 0])) for ct in
                                         TimetoString(sortedCommonTime, timeScale)] for districtData in
                                        allDistrictData]))


def getAllData(locationMain, update=True):
    # allFiles = [f[:-4] for f in listdir(locationMain) if isfile(join(locationMain, f))]
    allDistrictData, allDistrictMetaData = [], []

    for idx, row in getZones().sort_values('Zone', ascending=True).iterrows():
        print(row)
        file = join(locationMain + getCommonID(), row['Zone'] + '.txt')
        if update:
            data = [line.decode('unicode_escape')[:-1] for line in urlopen(
                join('http://berkeleyearth.lbl.gov/air-quality/maps/cities/' + row['Country'] + '/',
                     join(row['Division'].replace(' ', '_'), row['Zone'].replace(' ', '_') + '.txt')))]

            with open(file, 'w') as r:
                r.write('\n'.join(map(str, data)))


        else:
            data = str(
                open(file, 'rb').read().decode(
                    'unicode_escape')).split("\n")

        districtMetaData = np.array([d.split(':')[1][1:-1] for d in data[:9]])
        districtData = np.array([np.array(x.split('\t')) for x in np.array(data[10:-1])])
        allDistrictMetaData.append(districtMetaData)
        allDistrictData.append(districtData)

    return np.array(allDistrictData, dtype='object'), np.array(allDistrictMetaData)[:, [2, 4, 5, 6, 7]]


def LoadData():
    fname = berkely_earth_data + 'prepared/' + getCommonID() + '/reading.pickle'
    basicTimeParameters = np.array(
        ['year', 'month', 'day', 'hour', 'yearmonthday', 'yearmonthdayhour'])

    try:
        raise
        with open(fname, "rb") as f:
            dataSummaries, allDistrictMetaData = pk.load(f)
    except:
        allDistrictData, allDistrictMetaData = getAllData(berkely_earth_data + 'raw/')
        # dataSummaries = [getCommonTimes(allDistrictData, timeParam) for timeParam in basicTimeParameters]
        dataSummaries = getCommonTimes(allDistrictData, basicTimeParameters[-1])
        with open(fname, "wb") as f:
            pk.dump((dataSummaries, allDistrictMetaData), f)

    metaFrame = LoadMetadata(allDistrictMetaData)
    series = LoadSeries(dataSummaries)
    # print(metaFrame)
    # print(series)
    return dataSummaries


def LoadSeries(data=None, name='reading'):
    fname = berkely_earth_data_prepared + getCommonID() + '/timeseries'
    if data is not None:
        metaFrame = LoadMetadata()
        df = pd.DataFrame(data=np.transpose(data[1]), index=data[0],
                          columns=metaFrame.index).apply(pd.to_numeric)
        df = df.reset_index()
        df = df[sorted(df.columns.values)]
        df.to_feather(fname)

    return pd.read_feather(fname).set_index('index', drop=True)['2017':]
    # return pd.read_feather(fname).set_index('index', drop=True)['2017':].rename(
    #     columns={'Azimpur': 'Dhaka'}).sort_index(axis=1)


def LoadMetadata(allDistrictMetaData=None):
    metadataFileName = berkely_earth_data_prepared + getCommonID() + '/metadata'
    if allDistrictMetaData is not None:
        metaFrame = pd.DataFrame(columns=['Zone', 'Division', 'Population', 'Latitude', 'Longitude'],
                                 data=allDistrictMetaData).sort_values('Zone')
        metaFrame[['Population', 'Latitude', 'Longitude']] = metaFrame[['Population', 'Latitude', 'Longitude']].apply(
            pd.to_numeric).round(5)
        metaFrame.to_feather(metadataFileName)

    return pd.read_feather(metadataFileName).set_index('Zone', drop=True)
    # return pd.read_feather(metadataFileName).set_index('Zone', drop=True).rename(index={'Azimpur': 'Dhaka'}).sort_index(
    #     axis=0)


def makeHeaderFile(metaFrame):
    metaFrame = metaFrame.assign(Country='Bangladesh')
    metaFrame = metaFrame.reset_index()[['Country', 'Division', 'Zone']]
    metaFrame.to_csv('headerFile.csv', index=False)


def allZoneRead():
    alldatapath = berkely_earth_data + 'total/'
    allFiles = [alldatapath + f for f in listdir(alldatapath)]
    allDistrictMetaData = []

    for file in allFiles:
        data = str(open(file, 'rb').read().decode('unicode_escape')).split("\n")
        allDistrictMetaData.append(np.array([d.split(':')[1][1:-1] for d in data[:9]]))

    allDistrictMetaData = pd.DataFrame(np.array(allDistrictMetaData)[:, [4, 2]], columns=['Division', 'Zone']).assign(
        Country='Bangladesh')
    allDistrictMetaData = allDistrictMetaData[['Country', 'Division', 'Zone']]
    allDistrictMetaData.to_csv(berkely_earth_data + 'zones/allbd.csv', index=False)





if __name__ == '__main__':
    # dataSummaries = LoadData()
    timeseies = LoadSeries()
    metaFrame = LoadMetadata()


    # for i in range(len(dataSummaries[1])): print(metaFrame.iloc[i, 0],
    #                                                  np.count_nonzero(dataSummaries[1][i] == None))
    # deviatedList = sorted((np.std(dataSummaries[-1][1], axis=0)).argsort()[-27 * 15:][::-1])

    # metaFrame = metaFrame.assign(avgRead=np.mean(dataSummaries[-1][1][:,:],axis=1))
    # print(metaFrame[['Population','avgRead']].corr())
    # popYear = [157977153,159685424,161376708,163046161]

    exit()
