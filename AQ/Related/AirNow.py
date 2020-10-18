import csv
import statistics

import matplotlib.pyplot as plt
import numpy as np

# import plotly.express as px

Site, Parameter, DateLT, Year, Month, Day, Hour, NowCastConc, AQI, AQI_Category, Raw_Conc, ConcUnit, Duration, QCName = list(
    range(14))
id, date, time, sht_t, sht_h, pm1, pm25, pm10, co2 = list(range(9))

prompt = '1:All Statistics\n2:Individual Query\n3:Individual Statistic\n4:Year Data Plot'


def filePrepare():
    location = '/media/az/Study/Air Analysis/Dataset/AirNow_PM2.5_Dhaka/Yearly Data/'

    yearlyData = []
    for i in range(16, 20):
        filename = (location + 'Dhaka_PM2.5_20' + str(i) + '_YTD.csv')

        with open(filename) as csvfile:
            readCSV = np.array(list(csv.reader(csvfile, delimiter=',')))
            yearlyData.append(readCSV[np.where(readCSV[:, QCName] == 'Valid')])
            # [np.where(not readCSV[:, NowCastConc] == '-999')]

    return yearlyData


def GetYearData(yearlyData, year):
    print("Enter Year\n" + "\n".join(map(str, list(range(2016, 2020)))))
    return yearlyData[year - 2016]


def GetMonth(yearData, month):
    print("Enter Month (1-12)")
    return (yearData[np.where(yearData[:, Month].astype('int') == (month - 1))])[::24, ]


def getYear():
    print("Enter Year\n" + "\n".join(map(str, list(range(2016, 2020)))))
    return int(input())


def YearPlot(yearlyData):
    year = getYear()
    for month in range(12):
        yearData = GetYearData(yearlyData, year)
        monthData = GetMonth(yearData, month)
        plt.plot(monthData[:, Day], monthData[:, Raw_Conc].astype('float64'))
        plt.show()


def Statistics(yearlyData):
    print("Enter Year\n" + "\n".join(map(str, list(range(2016, 2020)))))
    year = int(input())
    print("Enter Month (1-12)")
    month = int(input())
    yearData = GetYearData(yearlyData, year)
    monthData = GetMonth(yearData, month)
    rawConcArray = (monthData[:, Raw_Conc].astype('float64'))
    print(rawConcArray)

    if rawConcArray is None or len(rawConcArray) == 0:
        return

    print('mean ', np.mean(rawConcArray))
    print('median ', np.median(rawConcArray))
    print('mode ', statistics.mode(rawConcArray))
    print('std ', np.std(rawConcArray))


def switcher(functions, func_name, vals):
    # functions = {
    #     1  : AllAnalysis,
    #     2  : IndividualQuery,
    #     3  : Statistics,
    #     4  : YearPlot
    # }

    active_function = functions.get(func_name)
    return active_function(vals)


def Plotting(X, Y):
    for y in Y: plt.plot(X, y)
    plt.show()


def IndividualQuery(yearlyData):
    year, month = int(input()), int(input())
    yearData = GetYearData(yearlyData, year)
    monthData = GetMonth(yearData, month)
    Plotting(monthData[:, Day], [monthData[:, Raw_Conc].astype('float64')])


def GetInput(functions):
    print("\n\n\nEnter Query")
    for k, v in functions.items(): print(k, v.__name__)


def AllAnalysis(yearlyData):
    for X in yearlyData:
        for i in range(12):
            monthly = (X[np.where(X[:, Month].astype('int') == (i + 1))])
            print(monthly)
            # print()


def SeasonalAnalysis(yearlyData):
    return


if __name__ == '__main__':

    functions = {
        1: AllAnalysis,
        2: IndividualQuery,
        3: Statistics,
        4: YearPlot,
        5: SeasonalAnalysis
    }

    yearlyData = filePrepare()
    query = -1

    while not query == 0:
        GetInput(functions)
        query = int(input())
        switcher(functions, query, yearlyData)

    exit()
