import numpy as np
from scipy import stats


def StasticalMeasures(subjects):
    while True:
        print()
        print("Enter Option for FMRI reading Statistics")
        print()
        print("1   Basic Statistics")
        print("2   T-Test")
        print("0   Return")

        option = int(input())

        if (option == 0): return

        print("Enter Person Number (1-9)")
        person = int(input()) - 1
        print("Enter Trial Number (1-360)")
        trial = int(input()) - 1
        data = subjects[person].dataset[trial]

        if (option == 1):
            mean = np.mean(data)
            st_dev = np.std(data)
            min = np.min(data)
            median = np.median(data)
            max = np.max(data)

            print("mean ", mean)
            print("Standard Deviation ", st_dev)
            print("Min ", min)
            print("Median ", median)
            print("Max ", max)

        if (option == 2):
            print("Enter T-Test value")
            tTestValue = int(input())
            print(stats.ttest_1samp(data, tTestValue))
