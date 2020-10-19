import csv

import matplotlib.pyplot as plt
import numpy as np

id, date, time, sht_t, sht_h, pm1, pm25, pm10, co2 = list(range(9))


def BarPlot(data):
    n_groups = len(data[0])
    labels = ['sht_t', 'sht_h', 'pm1', 'pm25', 'pm10']

    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.15
    opacity = 1

    # rects1 = plt.bar(index, means_frank, bar_width,
    #                  alpha=opacity,
    #                  color='b',
    #                  label='Frank')
    #
    # rects2 = plt.bar(index + bar_width, means_guido, bar_width,
    #                  alpha=opacity,
    #                  color='g',
    #                  label='Guido')

    for i, d in enumerate(data): _ = plt.bar(index + i * bar_width, d, bar_width, alpha=opacity, label=labels[i])

    plt.xlabel('Factors')
    plt.ylabel('Day of Month')
    plt.title('Factors by Day')
    plt.xticks(index + bar_width, range(len(data[0])))
    plt.legend()

    plt.tight_layout()
    plt.show()


def Plotting(Y, xlabel='', title='', x=None):
    if x is None: x = np.arange(len(Y[0])) + 1
    labels = ['sht_t', 'sht_h', 'pm1', 'pm25', 'pm10']
    # for y,label in zip(Y,labels): plt.plot(x,y,label=label,linestyle='-', marker='o')
    for y, label in zip(Y, labels): plt.plot(x, y, label=label, linestyle=' ', marker='o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Factors')
    plt.legend()
    plt.show()


def filePrepareDU():
    location = '/media/az/Study/Air Analysis/Dataset/DUDataMonth'

    daysData = []
    for i in range(2, 30):
        filename = (location + '/2019-10-' + f"{i:02d}" + '.csv')

        with open(filename) as csvfile:
            readCSV = np.array(list(csv.reader(csvfile, delimiter=',')))
            daysData.append(readCSV[1:])

    return np.array(daysData)


def GetDaysdataAverage(daysData):
    daysdataAverage = np.array(
        [np.mean(IndividualQueryHourBasedDay(daysData, day)[:, 1:], axis=0) for day in range(len(daysData))])
    # allDaysMean = np.mean(daysdataAverage,axis=0)
    return np.transpose(daysdataAverage)


def IndividualQueryHourBasedDay(daysData, day):
    singleDayData = daysData[day][1:]
    dayTimes, hourMeans = [], []
    for dayTime in singleDayData[:, time]:
        dayTimeSplit = dayTime.split(':')
        dayTimes.append(np.array([dayTimeSplit[0], dayTimeSplit[1], dayTimeSplit[2]]))
    dayTimes = np.array(dayTimes).astype('int')

    for u in np.unique(dayTimes[:, 0]): hourMeans.append(np.hstack(
        (u, np.mean((singleDayData[np.where(dayTimes[:, 0] == u)[0]])[:, sht_t:co2].astype('float64'), axis=0))))
    # Plotting(np.transpose(np.array(hourMeans)[:,1:]))
    return (np.array(hourMeans)[:, :])


def weekAnalysis(allDaysStat):
    # Plotting((allDaysStat))
    shp = np.shape(allDaysStat)
    weekSplit = np.reshape(allDaysStat, (shp[0], int(shp[1] / 7), 7))
    weekDayMean, weekMean = [(np.mean(weekSplit, axis=(i + 1))) for i in range(2)]
    weekDayName = ['Sat', 'Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri']

    Plotting((allDaysStat), xlabel='Date in Month')
    Plotting((weekMean), xlabel='Week of Month')
    Plotting((weekDayMean), xlabel='Weekday', x=weekDayName[3:] + weekDayName[:3])


def DayNightStat(oneDayData):
    print(oneDayData)
    Day = (oneDayData[(oneDayData[:, 0] >= 8) & (oneDayData[:, 0] < 20)])
    Night = (oneDayData[(oneDayData[:, 0] < 8) | (oneDayData[:, 0] >= 20)])
    print(Day)
    print(Night)
    Plotting(np.transpose(oneDayData[1:]), 'Hourly Reading')
    Plotting(np.transpose(Day[1:]), xlabel='Day Time Obsevation')
    Plotting(np.transpose(Night[1:]), xlabel='Night Time Obsevation')


if __name__ == '__main__':
    daysData = filePrepareDU()
    oneDayData = IndividualQueryHourBasedDay(daysData, 26)
    allDaysStat = GetDaysdataAverage(daysData)
    # weekAnalysis((allDaysStat))
    # DayNightStat(oneDayData)
    BarPlot(allDaysStat)

    exit()
