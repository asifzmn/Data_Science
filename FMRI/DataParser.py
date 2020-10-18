import numpy as np
import scipy.io


def DisplayVariable(subjects):
    while True:
        print()
        print("Enter option for displaying variables")
        print()
        print("1   File varibles with person number")
        print("2   Condition and corresponding words")
        print("0   Return")

        option = int(input())

        if (option == 1):

            print("Enter Person Number (1-9)")
            person = int(input()) - 1
            print()

            print("Enter Varible Number")
            items = ['__header__', '__version__', '__globals__', 'meta', 'info', 'data']
            for item in items:
                print(items.index(item), item)
            print()

            varible = int(input())
            if ((person >= 1 and person <= 9) and (varible >= 0 and varible <= 5)):
                print(subjects[person].mat[items[varible]])

        if (option == 2):
            for key, value in subjects[0].infoDictionary.conditionWord.items():
                print(key, ' ', value)

        if (option == 0): return


def SubjectGenerator(oneToNine):
    subjects = []
    for i in range(oneToNine):
        subjects.append(Subject(scipy.io.loadmat('DataSets/data-science-P' + str(i + 1) + '.mat')))
    return subjects


class Meta:
    def __init__(self, study, subject, ntrials, nvoxels, dimx, dimy, dimz, colToCoord, coordToCol):
        self.study = study
        self.subject = subject
        self.ntrials = ntrials
        self.nvoxels = nvoxels
        self.dimx = dimx
        self.dimy = dimy
        self.dimz = dimz
        self.colToCoord = colToCoord
        self.coordToCol = coordToCol


class InfoDictionary:
    def __init__(self, info):
        condNum = list(map(int, info[:, 1]))
        wordNum = list(map(int, info[:, 3]))
        word = info[:, 2]
        cond = info[:, 0]
        self.condition = {}
        self.word = {}
        self.conditionWord = {}

        for (k1, k2, v1, v2) in zip(condNum, wordNum, cond, word):
            self.condition.update({k1: v1})
            self.word.update({(k1, k2): v2})

        self.condition = dict(sorted(self.condition.items()))

        self.totalWords = []
        for key, cond in self.condition.items():
            wordList = []
            for wordNo in range(1, 6):
                wordList.append(self.word[(key, wordNo)])
            self.conditionWord.update({cond: wordList})
            self.totalWords.append(np.array(wordList))

        self.totalWords = np.array(self.totalWords)


class Subject:

    def __init__(self, mat):
        self.mat = mat
        self.header = mat['__header__']
        self.version = mat['__version__']
        self.globals = mat['__globals__']
        self.meta = self.PrepareMeta(mat['meta'])
        self.info = self.PrepareInfo(mat['info'])
        self.data = self.PrepareData(mat['data'])
        self.infoDictionary = self.PrepareInfoDictionary(self.info)

    def PrepareData(self, data):
        dataArr = []
        for entry in data:
            dataArr.append(entry[0][0])
        return np.array(dataArr)

    def PrepareInfoDictionary(self, preparedInfo):
        return InfoDictionary(np.array(preparedInfo))

    def PrepareInfo(self, info):
        infoArr = []
        for trailInfo in info[0]:
            cond = trailInfo[0][0]
            cond_number = trailInfo[1][0][0]
            word = trailInfo[2][0]
            word_number = trailInfo[3][0][0]
            epoch = trailInfo[4][0][0]

            singleTrialInfo = []
            singleTrialInfo.append(cond)
            singleTrialInfo.append(cond_number)
            singleTrialInfo.append(word)
            singleTrialInfo.append(word_number)
            singleTrialInfo.append(epoch)
            infoArr.append(singleTrialInfo)

        return np.array(infoArr)

    def PrepareMeta(self, meta):

        study = meta[0][0][0][0]  # 'science'
        subject = meta[0][0][1][0]  # 'P1'
        ntrials = meta[0][0][2][0][0]  # 360
        nvoxels = meta[0][0][3][0][0]  # 21764
        dimx = meta[0][0][4][0][0]  # 51
        dimy = meta[0][0][5][0][0]  # 61
        dimz = meta[0][0][6][0][0]  # 23
        colToCoord = np.array(meta[0][0][7])  # [21764x3 double]
        coordToCol = np.array(meta[0][0][8])  # [51x61x23 double]

        return Meta(study, subject, ntrials, nvoxels, dimx, dimy, dimz, colToCoord, coordToCol)
