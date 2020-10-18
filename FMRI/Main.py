from FMRI.Algortihm import TrainingAndTesting
from FMRI.DataParser import SubjectGenerator, DisplayVariable
from FMRI.Statistics import StasticalMeasures
from FMRI.Visualization import Visualization


def Statistics(subjects):
    return


if __name__ == '__main__':

    oneToNine = 9
    subjects = SubjectGenerator(oneToNine)
    # minL = 19750
    # Visualization(subjects)

    while True:
        print()
        print("Enter operation")
        print()
        print("1   Display varibles")
        print("2   Visualization of brain activity")
        print("3   Statistical analysis")
        print("4   Training models and testing")
        print("0   Exit")

        option = int(input())
        if (option == 0): exit()
        if (option == 1): DisplayVariable(subjects)
        if (option == 2): Visualization(subjects)
        if (option == 3): StasticalMeasures(subjects)
        if (option == 4): TrainingAndTesting(subjects)
