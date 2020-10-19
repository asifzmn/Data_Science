import os
import pandas as pd
from AirQuality.AQ_Analysis import PLotlyTimeSeries
from Electricity.ElecCommon import PlotUsageTimeseries, createAndSaveTimeSeries

mainpath = '/media/az/Study/Datasets/Electricity/DPDC data/'
segmentDirectory, unitDirectory = mainpath + 'District Data/segments', mainpath + 'District Data/units'

zones = ['ADABOR', 'NARINDA', 'AZIMPUR', 'TEJGAON', 'FATEULLAH', 'PARIBAG', 'KHILGAON', 'N.GONJ (EAST)', 'SHYAMOLI',
         'KAKRAIL', 'KAZLA', 'RAJARBAG', 'MATUAIL', 'POSTOGOLA', 'MANIKNAGAR', 'BANGSHAL', 'DEMRA', 'KAMRANGIRCHAR',
         'MOTIJHEEL', 'JIGATOLA', 'N.GONJ (WEST)', 'SHITALOKKHYA', 'RAMNA', 'SHYAMPUR', 'SWAMIBAG', 'JURAIN',
         'SIDDIRGONJ', 'LALBAG', 'MUGDAPARA', 'MOGHBAZAR', 'BANGLABAZAR', 'SATMASJID', 'BANOSRI', 'DHANMONDI',
         'BASHABO', 'SHERE B.NAGAR']

damaged = ['ADABOR', 'AZIMPUR', 'KHILGAON', 'N.GONJ (EAST)', 'SHYAMOLI', 'MATUAIL', 'N.GONJ (WEST)', 'SHITALOKKHYA',
           'SIDDIRGONJ', 'MUGDAPARA', 'SATMASJID', 'BASHABO']

cleanZones = list(filter(lambda x: x not in damaged, zones))


def basicInfo(df):
    # print(df.info())
    for c in df.columns.values: print(c, df[c].unique().shape)
    for c in df.columns.values: print(c, df[c].unique())
    print()


def dateSeries(x, date_rng): return x.set_index('MONTH', drop=True).reindex(date_rng)["KWH"]


def MonthlyUsage(predataNOCS):
    df = predataNOCS[predataNOCS['MONTH'].dt.month == 2][['MONTH', 'NOCS', 'KWH']]
    df = df.groupby(['NOCS', 'MONTH']).apply(lambda y: y.mean(axis=0))
    df = df.unstack().T.droplevel(level=0)['2018':]
    df.to_csv('Data Directory/NOCSELEC.csv')


def preparedistrictData2013_2016():
    # dflarges = pd.read_csv(mainpath + '2012-2016-kwh.csv', header=0,
    #                         low_memory=False, chunksize=4 * 10 ** 6)
    #
    # for i, dflarge in enumerate(dflarges):
    #     os.makedirs(os.path.join(segmentDirectory, str(i)))
    #     for df in dflarge.groupby('NOCS'): df[1].drop('NOCS', axis=1).set_index('CUSTOMER_NUM', drop=True).to_csv(
    #         os.path.join(segmentDirectory, str(i) + '/' + df[0]))

    for zone in zones[:]:
        ZoneDataList = []
        for i in range(16):
            file = os.path.join(segmentDirectory, str(i) + '/' + zone)
            if os.path.exists(file): ZoneDataList.append(pd.read_csv(file, parse_dates=[1]))
        df = pd.concat(ZoneDataList).drop_duplicates(['CUSTOMER_NUM', 'MONTH'])
        createAndSaveTimeSeries(df[['CUSTOMER_NUM', 'MONTH', 'KWH']], unitDirectory + "/" + zone)
        print(df)


def preparedistrictData2017_2019():
    predataNOCS = pd.read_csv('/media/az/Study/Datasets/Electricity/DPDC data/prepaid-lta-kwh.csv', sep=',', skiprows=0,
                              parse_dates=[1],
                              low_memory=False)
    print(predataNOCS.dtypes)
    print(predataNOCS['NOCS'].unique())
    createAndSaveTimeSeries(predataNOCS[['CUSTOMER_NUM', 'MONTH', 'KWH']], 'DPDC')
    df = pd.read_csv('DPDC', index_col='CUSTOMER_NUM', low_memory=False).T
    PlotUsageTimeseries(df, 'DPDC')


def UsageTS():
    for i, zone in enumerate(zones[30:]):
        df = pd.read_csv(unitDirectory + "/" + zone, index_col='CUSTOMER_NUM').T
        df.index = pd.to_datetime(df.index)
        PlotUsageTimeseries(df, zone)


def prepareforNTLFebruaryOnly():
    lst = []

    for zone in cleanZones:
        df = pd.read_csv(unitDirectory + "/" + zone, index_col='CUSTOMER_NUM').T
        df.index = pd.to_datetime(df.index)
        df = df[df.index.month == 2].mean(axis=1)
        # print(zone, df.isnull().any())
        lst.append(df)

    dfAll = pd.concat(lst, axis=1)
    dfAll.columns = cleanZones
    dfAll.to_csv('Data Directory/NOCSELEC2.csv')


def prepareforNTLYear():
    lst = []
    for zone in zones[:]:
        df = pd.read_csv(unitDirectory + "/" + zone, index_col='CUSTOMER_NUM').T
        df.index = pd.to_datetime(df.index)
        lst.append(df['2018'].median(axis=1))

    dfAll = pd.concat(lst, axis=1)
    dfAll.columns = zones
    print(dfAll)
    # print(dfAll.info())
    dfAll.to_csv('Data Directory/NOCSELEC2.csv')


if __name__ == '__main__':
    # preparedistrictData2013_2016()
    # preparedistrictData2017_2019()
    prepareforNTLYear()

    exit()
    # customerLoc = pd.read_csv('/media/az/Study/Datasets/Electricity/DPDC Azimpur Data/azimpur.csv', sep=',', skiprows=0)
    predata = pd.read_csv('/media/az/Study/Datasets/Electricity/DPDC Azimpur Data/gis-predata.csv', sep=',', skiprows=0)
    # postdata = pd.read_csv('/media/az/Study/Datasets/Electricity/DPDC Azimpur Data/gis-postdata.csv', sep=',', skiprows=0)

    # basicInfo(customerLoc)
    # basicInfo(postdata)
    # basicInfo(predata)

    predata = predata[["TOTAL_AMOUNT", "RECHARGE_BY"]]

    bins = [0, 500, 1000, 1500, 2000, 3000, 4000, 5000, 100000]
    predata['Group'] = pd.cut(predata['TOTAL_AMOUNT'], bins, right=False)
    df = pd.crosstab(predata['Group'], predata['RECHARGE_BY'])
    df.to_csv('dataElec.csv')
    # list = []
    # for index,row in df.iterrows():
    #     dict = {"categorie":str(index.left)+"-"+str(index.right)}
    #     dict["values"] = [{"value": r,"rate": i} for i,r in row.iteritems()]
    #     list.append(dict)
    # with open('Electricity/dataElec.json', 'w') as outfile:json.dump(list,outfile)

    # a = predata['CUSTOMER_NUM']//10
    # b = postdata['CUSTOMER_NUM']
    # c =  customerLoc['CUSTOMER_NUM']//10
    #
    # print(a.unique().shape,b.unique().shape,c.unique().shape)
    # print(np.shape(np.intersect1d(a,b)))
    # print(np.shape(np.union1d(a,b)))
    # print(np.shape(np.intersect1d(b,c)))
    # print(np.shape(np.intersect1d(c,a)))

    # print(predata.columns.values)
    # print(predata.CUSTOMER_NUM.unique().shape)
    # print(predata.METER_NO.unique().shape)
    # print(predata.TARIFF.unique())
    # print(predata.LOCATION_CODE.unique())
    # print(predata.NOCS_NAME.unique())
    # print(predata.RECHARGE_BY.unique())

    exit()
