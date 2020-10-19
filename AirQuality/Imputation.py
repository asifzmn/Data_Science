import pickle
from collections import Counter
from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import LSTM, Dense, Masking, Dropout

from AirQuality.DataPreparation import LoadData


def BasicRNNImputation():
    df = pd.DataFrame(np.transpose(dataSummaries[1])[:, 0])
    df = pd.concat([df.shift(1), df], axis=1)
    df.fillna(0.0, inplace=True)
    values = df.values
    X, y = values, values[:, 0]
    X = X.reshape(len(X), 2, 1)
    y = y.reshape(len(y), 1)

    name = "prediction1.pickle"
    try:
        with open('', "rb") as f:
            yhat = pickle.load(f)
    except:
        model = Sequential()
        model.add(Masking(mask_value=0.0, input_shape=(2, 1)))
        model.add(LSTM(9, input_shape=(2, 1)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(X, y, epochs=5, batch_size=1, verbose=2)

        yhat = model.predict(X)
        with open(name, "wb") as f:
            pickle.dump(yhat, f)
        # with open(name, "wb") as f:
        #     pickle.dump((dataSummaries, allDistrictMetaData), f)

    for i in range(len(X)):
        if (y[i, 0] == 0.0): print('Expected', y[i, 0], 'Predicted', yhat[i, 0])

    df1 = pd.DataFrame(np.transpose(dataSummaries[1])[:, 0])
    df2 = pd.DataFrame(np.transpose(yhat)[0])
    # df = pd.concat([df1, df2], axis=1)

    # sns.set(style='darkgrid')
    # plt.figure(figsize=(200, 5))
    # df2.plot(style='.')
    # plt.show()


def Gaps():
    for i in range(22):
        singleDistrictData = dataSummaries[-1][i, 20000:]
        idx_pairs = np.where(np.diff(np.hstack(([False], singleDistrictData == None, [False]))))[0].reshape(-1, 2)
        counts = ((Counter(idx_pairs[:, 1] - idx_pairs[:, 0])))
        sortedCounts = dict(sorted(counts.items()))
        print(sortedCounts.popitem())


def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(look_back, len(dataset) - look_back - 1):
        X.append(dataset[(i - look_back):(i + look_back), 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)


if __name__ == '__main__':
    locationMain = '/media/az/Study/Air Analysis/Dataset/Berkely Earth Dataset/'
    allFiles = [f for f in listdir(locationMain) if isfile(join(locationMain, f))]
    basicTimeParameters = np.array(
        ['year', 'month', 'day', 'hour', 'yearmonth', 'monthday', 'yearmonthday', 'yearmonthdayhour'])

    dataSummaries, allDistrictMetaData = LoadData(allFiles, basicTimeParameters, 'reading2.pickle')
    dataSummaries = dataSummaries[-1]

    # print(pd.DataFrame(np.transpose(dataSummaries[1])).isnull().sum())
    # print(np.shape(dataSummaries[1][0]))
    # print(np.shape(np.where(np.transpose(dataSummaries[1][0]) == None)))

    solidDataLoc = (np.where(np.any(dataSummaries[1] == None, axis=0) == False))[0]
    if not len(solidDataLoc) == 0: dataSummaries = dataSummaries[0][solidDataLoc[0]:solidDataLoc[-1]], dataSummaries[1][
                                                                                                       :,
                                                                                                       solidDataLoc[0]:
                                                                                                       solidDataLoc[-1]]

    df = pd.DataFrame(np.transpose(dataSummaries[1])[:, 5])
    df.fillna(0.0, inplace=True)

    # np.random.seed(0)
    # print(np.linspace(2,3,3+2)[1:-1] + np.random.randn(3) * np.std([2,3]))
    # print(np.random.rand(3))
    # exit()

    dataset = df.values  # numpy.ndarray
    dataset = dataset.astype('float32')
    # dataset = np.reshape(dataset, (-1, 1))
    dataset = np.reshape(dataset, (-1, 1))[:, 0]
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # dataset = scaler.fit_transform(dataset)[:,0]
    # print(np.shape(dataset))
    # print('---')
    # print(np.shape(dataset[np.where(dataset!=0.0)]))
    # print(np.shape(dataset[np.where(dataset == 0.0)]))

    look_back = 24 * 7
    singleDistrictData = dataset
    idx_pairsX = np.where(np.diff(np.hstack(([False], dataset != 0.0, [False]))))[0].reshape(-1, 2)
    # idx_pairsX = np.where(np.diff(np.hstack(([True], dataset != 0.0, [True]))))[0].reshape(-1, 2)
    # idx_pairsY = np.where(np.diff(np.hstack(([True], dataset == 0.0, [True]))))[0].reshape(-1, 2)
    idx_pairsY = np.where(np.diff(np.hstack(([False], dataset == 0.0, [False]))))[0].reshape(-1, 2)
    diffsX = idx_pairsX[:, 1] - idx_pairsX[:, 0]
    diffsY = idx_pairsY[:, 1] - idx_pairsY[:, 0]
    print(np.shape(idx_pairsX))
    print(np.shape(idx_pairsY))
    print(Counter(diffsX))
    print(Counter(diffsY))
    diffsMinValidsX = (idx_pairsX[diffsX > look_back])
    diffsMinValidsY = (idx_pairsY[diffsY > look_back])

    X, Y = [], []
    for dif in diffsMinValidsX:
        for i in range(look_back + dif[0], dif[1]):
            # print(dataset[i - look_back])
            X.append(dataset[(i - look_back):(i)])
            Y.append(dataset[i])

    X, Y = np.array(X), np.array(Y)
    print(np.shape(X))
    print(np.shape(Y))

    print(sorted(Counter(diffsX)))
    print(sorted(Counter(diffsY)))

    smallGaps = ((idx_pairsY[diffsY <= 3]))
    # for i in range(len(smallGaps)-1):dataset[smallGaps[i][0]:smallGaps[i][1]] = np.mean(dataset[[smallGaps[i][0]-1,smallGaps[i][1]]])
    for gap in smallGaps[:-1]: dataset[gap[0]:gap[1]] = np.linspace(dataset[gap[0] - 1], dataset[gap[1]],
                                                                    gap[1] - gap[0] + 2)[1:-1]

    idx_pairsX = np.where(np.diff(np.hstack(([False], dataset != 0.0, [False]))))[0].reshape(-1, 2)
    idx_pairsY = np.where(np.diff(np.hstack(([False], dataset == 0.0, [False]))))[0].reshape(-1, 2)
    diffsX = idx_pairsX[:, 1] - idx_pairsX[:, 0]
    diffsY = idx_pairsY[:, 1] - idx_pairsY[:, 0]
    print(np.shape(idx_pairsX))
    print(np.shape(idx_pairsY))
    print(Counter(diffsX))
    print(Counter(diffsY))
    diffsMinValidsX = (idx_pairsX[diffsX > look_back])
    diffsMinValidsY = (idx_pairsY[diffsY > look_back])

    X, Y = [], []
    for dif in diffsMinValidsX:
        for i in range(look_back + dif[0], dif[1]):
            # print(dataset[i - look_back])
            X.append(dataset[(i - look_back):(i)])
            Y.append(dataset[i])

    X, Y = np.array(X), np.array(Y)
    print(np.shape(X))
    print(np.shape(Y))
    # print((Counter(diffsY)))
    print(sorted(Counter(diffsX)))
    print(sorted(Counter(diffsY)))
    # diffsMinValidsX = (idx_pairsX[diffsX > look_back])
    # diffsMinValidsY = (idx_pairsY[diffsY > look_back])
    exit()
    # print(diffsMinValids)
    # print(sortedCounts.popitem())
    print(np.shape(dataset))
    X, Y = [], []
    for dif in diffsMinValidsX:
        for i in range(look_back + dif[0], dif[1]):
            # print(dataset[i - look_back])
            X.append(dataset[(i - look_back):(i)])
            Y.append(dataset[i])

    X, Y = np.array(X), np.array(Y)
    print(np.shape(X))
    print(np.shape(Y))

    X, Y = [], []
    for dif in diffsMinValidsX:
        for i in range(dif[0], dif[1] - look_back):
            # print(dataset[i - look_back])
            X.append(dataset[(i + 1):(i + look_back + 1)])
            Y.append(dataset[i])

    X_train, Y_train = np.array(X), np.array(Y)
    print(np.shape(X))
    print(np.shape(Y))
    # print(X[0],Y[0])
    # print(X[1],Y[1])
    # print(X[2],Y[2])

    # exit()
    #
    # dataset = df.values  # numpy.ndarray
    # dataset = dataset.astype('float32')
    # dataset = np.reshape(dataset, (-1, 1))
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # dataset = scaler.fit_transform(dataset)
    # train_size = int(len(dataset) * 0.80)
    # test_size = len(dataset) - train_size
    # train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

    # look_back = 300
    # X_train, Y_train = create_dataset(train, look_back)
    # X_test, Y_test = create_dataset(test, look_back)

    # reshape input to be [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    # X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    model = Sequential()
    model.add(Masking(mask_value=0.0, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # history = model.fit(X_train, Y_train, epochs=10, batch_size=70, validation_data=(X_test, Y_test),
    #                     callbacks=[EarlyStopping(monitor='val_loss', patience=10)], verbose=1, shuffle=False)

    history = model.fit(X_train, Y_train, epochs=10, batch_size=70,
                        callbacks=[EarlyStopping(monitor='val_loss', patience=10)], verbose=1, shuffle=False)

    model.summary()

    # train_predict = model.predict(X_train)
    # test_predict = model.predict(X_test)
    # # invert predictions
    # train_predict = scaler.inverse_transform(train_predict)
    # Y_train = scaler.inverse_transform([Y_train])
    # test_predict = scaler.inverse_transform(test_predict)
    # Y_test = scaler.inverse_transform([Y_test])
    # print('Train Mean Absolute Error:', mean_absolute_error(Y_train[0], train_predict[:, 0]))
    # print('Train Root Mean Squared Error:', np.sqrt(mean_squared_error(Y_train[0], train_predict[:, 0])))
    # print('Test Mean Absolute Error:', mean_absolute_error(Y_test[0], test_predict[:, 0]))
    # print('Test Root Mean Squared Error:', np.sqrt(mean_squared_error(Y_test[0], test_predict[:, 0])))
    #
    # plt.figure(figsize=(8, 4))
    # plt.plot(history.history['loss'], label='Train Loss')
    # plt.plot(history.history['val_loss'], label='Test Loss')
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epochs')
    # plt.legend(loc='upper right')
    # plt.show()
    #
    # a = len(Y_test[0])
    # aa = [x for x in range(a)]
    # plt.figure(figsize=(8, 4))
    # plt.plot(aa, Y_test[0][:a], marker='.', label="actual")
    # plt.plot(aa, test_predict[:, 0][:a], 'r', label="prediction")
    # # plt.tick_params(left=False, labelleft=True) #remove ticks
    # plt.tight_layout()
    # sns.despine(top=True)
    # plt.subplots_adjust(left=0.07)
    # plt.ylabel('PM', size=15)
    # plt.xlabel('Time step', size=15)
    # plt.legend(fontsize=15)
    # plt.show()

    # TimeFrame = pd.DataFrame(pd.to_datetime(['/'.join(map(str, d[:-1])) for d in dataSummaries[0]])) + pd.DataFrame( data=dataSummaries[0][:, -1].astype('int') * 3600 * 1000000000)
    # df = pd.DataFrame(data=np.transpose(dataSummaries[1]), index=TimeFrame[0].values, columns=[d.replace(' ', '') for d in allDistrictMetaData[:, 0]])

    # df = df['Dhaka']
    # df = df['2018']

    # print(df.isnull().sum())
    exit()
