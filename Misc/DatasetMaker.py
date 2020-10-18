import numpy as np
import pandas as pd

if __name__ == '__main__':
    dataCount, daySupportRange = 1000, 21
    slumInfo, organizationinfo = pd.read_csv('/home/az/Desktop/slum_summary.csv', sep=',', skiprows=0), pd.read_csv(
        '/home/az/Desktop/demo_org_info - org.csv', sep=',', skiprows=0)
    organizationArray = organizationinfo.values
    slumArray = slumInfo[['Slum_ID', 'Slum/Colony Name', 'Number of Households']].values
    datedata = np.arange('2020-03-29', '2020-04-03', dtype='datetime64[D]')
    organization = organizationArray[np.random.choice(organizationArray.shape[0], dataCount)]
    slum = slumArray[np.random.choice(slumArray.shape[0], dataCount)]
    date = (np.random.choice(datedata, dataCount)).reshape(-1, 1)
    daySupport = (np.random.randint(1, daySupportRange, dataCount)).reshape(-1, 1)
    HouseCover = (slum[:, 2] * np.random.uniform(low=0.1, high=.9, size=(dataCount,))).astype('int').reshape(-1, 1)
    columnsList = 'Organization ID', 'Organization Name', 'Slum ID', 'Slum Name', 'Date in which the aid was provided', 'Number of Households Covered in that slum', 'Number of days the aid will last'
    df = pd.DataFrame(data=np.hstack((organization, slum[:, :2], date, HouseCover, daySupport)), columns=columnsList)
    df.index.name = 'Aid ID'
    df.to_csv('sampleData1000.csv', sep="*")

    dff = (pd.read_csv('sampleData1000.csv', sep="*")).values
    for d in dff: print(d)
    print(dff.shape)
