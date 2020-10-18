import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import butter, filtfilt, hilbert
from scipy.stats import stats


def PearsonCorr(df):
    # df = pd.read_csv('synchrony_sample.csv')
    overall_pearson_r = df.corr().iloc[0, 1]
    print(f"Pandas computed Pearson r: {overall_pearson_r}")
    # out: Pandas computed Pearson r: 0.2058774513561943

    r, p = stats.pearsonr(df.dropna().iloc[0], df.dropna().iloc[0])
    print(f"Scipy computed Pearson r: {r} and p-value: {p}")
    # out: Scipy computed Pearson r: 0.20587745135619354 and p-value: 3.7902989479463397e-51

    # Compute rolling window synchrony
    f, ax = plt.subplots(figsize=(7, 3))
    df.rolling(window=30, center=True).median().plot(ax=ax)
    ax.set(xlabel='Time', ylabel='Pearson r')
    ax.set(title=f"Overall Pearson r = {np.round(overall_pearson_r, 2)}")

    plt.show()


def crosscorr(datax, datay, lag=0, wrap=False):
    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)

    # print(datax.dtype())

    return datax.corr(datay.shift(lag))


def CrossCorr(df):
    # r_window_size = 120
    # # Interpolate missing data.
    # # df_interpolated = df.interpolate()
    # # Compute rolling window synchrony
    # rolling_r = df.iloc[:,0].rolling(window=r_window_size, center=True).corr(df.iloc[:,1])
    # f, ax = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    # df.rolling(window=30, center=True).median().plot(ax=ax[0])
    # ax[0].set(xlabel='Frame', ylabel='Smiling Evidence')
    # rolling_r.plot(ax=ax[1])
    # ax[1].set(xlabel='Frame', ylabel='Pearson r')
    # plt.suptitle("Smiling data and rolling window correlation")
    # plt.show()

    d1 = df.iloc[:, 0]
    d2 = df.iloc[:, 1]
    # seconds = 5
    # fps = 3
    lagRange = 15
    rs = [crosscorr(d1, d2, lag) for lag in range(-lagRange, lagRange + 1)]
    offset = np.floor(len(rs) / 2) - np.argmax(rs)
    f, ax = plt.subplots(figsize=(14, 3))
    print(offset)

    ax.plot(rs)
    ax.axvline(np.ceil(len(rs) / 2), color='k', linestyle='--', label='Center')
    ax.axvline(np.argmax(rs), color='r', linestyle='--', label='Peak synchrony')
    # ax.set(title=f'Offset = {offset} frames\nS1 leads <> S2 leads', ylim=[.1, .31], xlim=[0, 301], xlabel='Offset',
    ax.set(title=f'Offset = {offset} frames\nS1 leads <{max(rs)}> S2 leads', xlabel='Offset',
           ylabel='Pearson r')
    # ax.set_xticks([0, 50, 100, 151, 201, 251, 301])
    # ax.set_xticklabels([-150, -100, -50, 0, 50, 100, 150])
    plt.legend()
    plt.show()


def Windowed(df):
    # seconds = 3
    # fps = 1
    # # no_splits = 3*24
    # # samples_per_split = int(df.shape[0] / no_splits)
    # samples_per_split = 24
    # no_splits = int(df.shape[0] / samples_per_split)
    # rss = []
    # for t in range(0, no_splits):
    #     d1 = df['Dhaka'].iloc[(t) * samples_per_split:(t + 1) * samples_per_split]
    #     d2 = df['Narsingdi'].iloc[(t) * samples_per_split:(t + 1) * samples_per_split]
    #     # print(d1)
    #     rs = [crosscorr(d1, d2, lag) for lag in range(-int(seconds * fps), int(seconds * fps + 1))]
    #     rss.append(rs)
    # rss = pd.DataFrame(rss)
    # f, ax = plt.subplots(figsize=(3, 100))
    # sns.heatmap(rss, cmap=(sns.diverging_palette(15, 250, s=90, l=50, n=90, center="light")), ax=ax,vmin = -1, vmax = 1)
    # ax.set(title=f'Windowed Time Lagged Cross Correlation', xlim=[0, 2*seconds+1], xlabel='Offset', ylabel='Window epochs')
    # # ax.set(title=f'Windowed Time Lagged Cross Correlation',  xlabel='Offset', ylabel='Window epochs')
    # # ax.set_xticks([0, 50, 100, 151, 201, 251, 301])
    # # ax.set_xticklabels([-150, -100, -50, 0, 50, 100, 150])
    # plt.show()

    seconds = 3
    fps = 1
    window_size = 300  # samples
    t_start = 0
    t_end = t_start + window_size
    step_size = 30
    rss = []
    while t_end < df.shape[0]:
        d1 = df['Dhaka'].iloc[t_start:t_end]
        d2 = df['Narsingdi'].iloc[t_start:t_end]
        rs = [crosscorr(d1, d2, lag, wrap=False) for lag in range(-int(seconds * fps), int(seconds * fps + 1))]
        rss.append(rs)
        t_start = t_start + step_size
        t_end = t_end + step_size
    rss = pd.DataFrame(rss)

    f, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(rss, cmap=sns.diverging_palette(15, 250, s=90, l=50, n=90, center="light"), ax=ax)
    # ax.set(title=f'Rolling Windowed Time Lagged Cross Correlation', xlim=[0, 301], xlabel='Offset', ylabel='Epochs')
    ax.set(title=f'Rolling Windowed Time Lagged Cross Correlation', xlabel='Offset', ylabel='Epochs')
    # ax.set_xticks([0, 50, 100, 151, 201, 251, 301])
    # ax.set_xticklabels([-150, -100, -50, 0, 50, 100, 150]);
    plt.show()


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def instantaneous_phase_synchrony(df):
    lowcut = .01
    highcut = .5
    fs = 30.
    order = 1
    d1 = df['Dhaka'].iloc[:1000].interpolate().values
    d2 = df['Narsingdi'].iloc[:1000].interpolate().values
    y1 = butter_bandpass_filter(d1, lowcut=lowcut, highcut=highcut, fs=fs, order=order)
    y2 = butter_bandpass_filter(d2, lowcut=lowcut, highcut=highcut, fs=fs, order=order)

    al1 = np.angle(hilbert(y1), deg=False)
    al2 = np.angle(hilbert(y2), deg=False)
    phase_synchrony = 1 - np.sin(np.abs(al1 - al2) / 2)
    N = len(al1)

    # Plot results
    f, ax = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
    ax[0].plot(y1, color='r', label='y1')
    ax[0].plot(y2, color='b', label='y2')
    ax[0].legend(bbox_to_anchor=(0., 1.02, 1., .102), ncol=2)
    ax[0].set(xlim=[0, N], title='Filtered Timeseries Data')
    ax[1].plot(al1, color='r')
    ax[1].plot(al2, color='b')
    ax[1].set(ylabel='Angle', title='Angle at each Timepoint', xlim=[0, N])
    # phase_synchrony = 1 - np.sin(np.abs(al1 - al2) / 2)
    ax[2].plot(phase_synchrony)
    ax[2].set(ylim=[0, 1.1], xlim=[0, N], title='Instantaneous Phase Synchrony', xlabel='Time',
              ylabel='Phase Synchrony')
    plt.tight_layout()
    plt.show()
