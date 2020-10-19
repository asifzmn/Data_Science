from itertools import combinations
from os import listdir
from os.path import isfile, join

import numpy as np

[division, district, latt, long, Year, Month, Day, UTC_Hour, PM25, PM10_mask, Retrospective] = range(11)


def GetDataBase(allFiles):
    allDistrictData, allDistrictMetaData = [], []
    for file in allFiles[:3]:
        data = str(open(join(locationMain, file), 'rb').read().decode('unicode_escape')).split("\n")
        districtMetaData = np.array([d.split(':')[1][:-1] for d in data[:9]])
        districtData = np.array(
            [np.hstack((districtMetaData[[4, 2]], x.split('\t')))[:-1] for x in np.array(data[10:-1])])
        allDistrictData.append(districtData)

    database = np.vstack(allDistrictData)
    np.random.shuffle(database)

    divTable = (np.array([[i, div] for i, div in enumerate(np.unique(database[:, division]))]))
    divTableDict = dict([div, i] for i, div in divTable)
    disTable = (np.array([[i, dis[1], divTableDict[dis[0]]] for i, dis in
                          enumerate(np.unique(database[:, division:district + 1], axis=0))]))
    disTableDict = dict([dis, [i, divID]] for i, dis, divID in disTable)
    database = np.array([np.hstack((disTableDict[data[1]][0], data[2:])) for data in database])

    return database


def GetDataCube(database):
    districtSlices = []
    for districtId in np.unique(database[:, 0]):
        districtData = database[(np.where(database[:, 0] == districtId))]
        uniqueTime = (
            np.array([np.hstack((i, time)) for i, time in enumerate(np.unique(districtData[:, 1:3], axis=0))]))
        # uniqueTimeDict = dict([ut[0],ut[1:]] for ut in uniqueTime)
        districtSlices.append(np.array([(np.mean(
            np.array([data[-2:].astype('float64') for data in districtData if np.array_equal(data[1:3], date)]),
            axis=0)) for date in uniqueTime[:, 1:3]]))
    return (np.array(districtSlices))


def Slice(dataCube, axis, index): return dataCube.take(indices=index, axis=axis)


def Dice(dataCube, indeices, axis=0): return dataCube if np.ndim(dataCube) == axis else Dice(
    dataCube.take(indices=indeices[axis], axis=axis), indeices, axis + 1)


def LatticeodCuboid(dataCube): return [
    [np.mean(dataCube, axis=perm) for perm in list(combinations(range(np.ndim(dataCube)), i + 1))] for i in
    range(np.ndim(dataCube))]


if __name__ == '__main__':
    locationMain = '/media/az/Study/Air Analysis/Dataset/Berkely Earth Dataset/'
    allFiles = [f for f in listdir(locationMain) if isfile(join(locationMain, f))]
    database = GetDataBase(allFiles)
    # print("\n\n\n\n\n\n---------------------------------------------")
    dataCube = GetDataCube(database)
    print(dataCube)
    print("\n\n\n\n\n\n---------------------------------------------")
    dataCubioid = LatticeodCuboid(dataCube)
    print('\n\n\n'.join(map(str, dataCubioid)))
    print("\n\n\n\n\n---------------------------------------------")

    inputnum = int(input())
    while not inputnum == 0:
        if inputnum == 1:
            print('Enter Slicing Axis and Index ')
            sliced = Slice(dataCube, int(input()), int(input()))
            print(sliced)
            if int(input()) == 1:
                print(np.rot90(sliced))
        if inputnum == 2:
            print('Enter Dicing Indices For Each Axis ')
            # print(Dice(dataCube,[[1,2],[1,3,6,17],[0]]))
            print(Dice(dataCube,
                       [np.array(input().split(' ')).astype('int'), np.array(input().split(' ')).astype('int'),
                        np.array(input().split(' ')).astype('int')]))
        inputnum = int(input())

    exit()
