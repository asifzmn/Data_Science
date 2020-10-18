# from bs4 import BeautifulSoup
# import urllib.request
# import nltk
# from nltk.corpus import stopwords
#
# response = urllib.request.urlopen('http://php.net/')
# html = response.read()
# soup = BeautifulSoup(html,"html.parser")
# text = soup.get_text(strip=True)
# tokens = [t for t in text.split()]
# clean_tokens = tokens[:]
# sr = stopwords.words('english')
# print(len(sr))
# for token in tokens:
#     if token in sr:
#         clean_tokens.remove(token)
# freq = nltk.FreqDist(clean_tokens)
# for key,val in freq.items():
#     print (str(key) + ':' + str(val))
# freq.plot(20, cumulative=False)

#
# from nltk.corpus import wordnet
# syn = wordnet.synsets("finished")
# print(syn[0].definition())
# print(syn[0].examples())
#
# from nltk.corpus import wordnet
# synonyms = []
# for syn in wordnet.synsets('food'):
#     for lemma in syn.lemmas():
#         synonyms.append(lemma.name())
# print(synonyms)

# from nltk.stem import PorterStemmer
# from nltk.tokenize import sent_tokenize, word_tokenize
#
# words = ["game", "gaming", "gamed", "games"]
# ps = PorterStemmer()
#
# for word in words:
#     print(ps.stem(word))

# from nltk.stem import PorterStemmer
# from nltk.tokenize import sent_tokenize, word_tokenize
#
# ps = PorterStemmer()
#
# sentence = "gaming, the gamers play games"
# words = word_tokenize(sentence)
#
# for word in words:
#     print(word + ":" + ps.stem(word))


# import nltk
# from nltk.tokenize import PunktSentenceTokenizer
#
# document = 'Whether you\'re new to programming or an experienced developer, it\'s easy to learn and use Python.'
# sentences = nltk.sent_tokenize(document)
# for sent in sentences:
#     print(nltk.pos_tag(nltk.word_tokenize(sent)))
# import nltk
# from nltk.corpus import names
#
# def gender_features(word):
#     return {'last_letter': word[-1]}
#
#
# if __name__ == '__main__':
#
#     names = ([(name, 'male') for name in names.words('male.txt')] +
#              [(name, 'female') for name in names.words('female.txt')])
#
#     print(names)
#
#     train_set = [(gender_features(n), g) for (n,g) in names]
#     #
#     # m = 0
#     # f = 0
#     # for train_set_element in train_set:
#     #     if(train_set_element[1]=='male' and train_set_element[0].get('last_letter')=='k'):
#     #         m+=1
#     #     elif(train_set_element[0]=='female' and train_set_element[0].get('last_letter')=='k'):
#     #         f+=1
#     #
#     # print(m)
#     # print(f)
#
#     classifier = nltk.NaiveBayesClassifier.train(train_set)
#
#     print(classifier.classify(gender_features('Frank')))
# import csv
#
# with open('/home/az/Downloads/Datasets/trainingandtestdata/training.1600000.processed.noemoticon.csv') as csv_file:
#     csv_reader = csv.reader(csv_file, delimiter=',')
#     line_count = 0
#     for row in csv_reader:
#         print(line_count)
#         line_count += 1
#     print(f'Processed {line_count} lines.')
#
# import pandas as pd
# train = pd.read_csv('/home/az/Downloads/Datasets/trainingandtestdata/training.1600000.processed.noemoticon.csv')
# print(len(train))
#
# import pandas as pd
# chunksize = 10 ** 8
# a=0
# for chunk in pd.read_csv('/home/az/Downloads/Datasets/trainingandtestdata/training.1600000.processed.noemoticon.csv', chunksize=chunksize):
#     print(a)
#     a+=1
# data-science-P1.mat
#
#
# __header__ ----------------------------------- b'MATLAB 5.0 MAT-file, Platform: GLNXA64, Created on: Tue Oct 20 19:12:48 2009'
# .............
# __version__ ----------------------------------- 1.0
# .............
# __globals__ ----------------------------------- []
# .............
#   meta
#   info
#   data

# print(self.header)
# print()
# print(self.version)
# print()
# print(self.globals)
# print()
# print(mat['meta'])
# print()
# print(self.info)
# print()
# print(self.data)
# print(np.shape(mat['meta']))
# print(np.shape(mat['info']))
# print(np.shape(mat['data']))


#
# def euclidean_distance(x, y):
#     return sqrt(sum(pow(a - b, 2) for a, b in zip(x, y)))
#
#
# def similarity():
#     for targetWord in ['tool', 'hammer'][0:1]:
#         targetoccurances = []
#         for x in range(len(info)):
#             if targetWord in info[x]:
#                 targetoccurances.append(x)
#
#         # print(targetoccurances)             # [49, 69, 158, 195, 250, 323]
#         # print(len(targetoccurances))
#         # for y in targetoccurances:
#         #     print(data[y])
#
#         perm = permutations(targetoccurances, 2)
#
#         sim = Similarity()
#         simList = []
#
#         for itr in list(perm):
#             a = data[itr[0]]
#             b = data[itr[1]]
#             c = sim.cosine_similarity(a, b)
#             simList.append(c)
#
#         print(i)
#         print(simList)
#         print(max(simList))
#         print(min(simList))
#         print(max(simList) - min(simList))
#         print()
#
#         d=data
#         print(sim.euclidean_distance(d[49],d[158]))
#         print(sim.manhattan_distance(d[49],d[158]))
#         print(sim.minkowski_distance(d[49],d[158]))
#         print(sim.cosine_similarity(d[49],d[69]))
#         print(sim.cosine_similarity(d[49],d[158]))
#         print(sim.cosine_similarity(d[49],d[195]))
#         print(sim.cosine_similarity(d[49],d[250]))
#         print(sim.cosine_similarity(d[49],d[323]))
#         print(sim.jaccard_similarity(d[49],d[158]))

# Nothng significant in similarity tests among same condition or word occurences

# import scipy.io
#
# if __name__ == '__main__':
#
#     mat = scipy.io.loadmat('data-science-P1.mat')
#
#     a = mat['meta']
#     b = a[0][0]

#   print(a)
# print('-------------------------------------------------------------------')
# print(len(a[0][0]))
# for x in a[0][0]:
#     print(x)
#     print('_______________________')

# print(len(b[7]))
# for x in b[7]:
#     print(x)

#
# oneToNine = 1
# subjects = SubjectGenerator(oneToNine)

# a = ''
# for i in range(20000):a=a+'a'
# print(a)

# for subjectItr in range(oneToNine):
#     subject = subjects[subjectItr]
#     meta = subject.meta
#     info = subject.info
#     data = subject.data

# print(meta.colToCoord[3])
# print(data[:,3])
# print(data[0])
# print(np.shape(meta.colToCoord))
# print(np.shape(meta.coordToCol))
# print(meta.dimx)
# print(meta.dimy)
# print(meta.dimz)
# print(meta.ntrials)
# print(meta.nvoxels)
# print()

# print(meta.colToCoord[382])
# print(meta.coordToCol[31][10][1])

# coordtoColumnDictionary = {}
# s = []

# for x in range(meta.dimx):
#     for y in range(meta.dimy):
#         for z in range(meta.dimz):
#             if meta.coordToCol[x][y][z]!=0 :
#                 print(x,y,z,meta.coordToCol[x][y][z])
#                 coordtoColumnDictionary.update({(x,y,z):meta.coordToCol[x][y][z]})
#                 if(meta.coordToCol[x][y][z]==1):
#                     s.append(x)
#                     s.append(y)
#                     s.append(z)

# print(s)
# print(meta.colToCoord[1])


# for x in range(1):
#     print(meta.coordToCol[14+x][12][1])
#
# for w in range(15):
#     print(meta.colToCoord[w])

# [31 10  1] 382
# [32 10  1] 383
# [33 10  1] 384
# [34 10  1] 385
# [35 10  1] 386
# [29 11  1] 406
# [30 11  1] 407
# [31 11  1] 408
# [32 11  1] 409
# [33 11  1] 410
# [34 11  1] 411
# [35 11  1] 412
# [36 11  1] 413
# [37 11  1] 414
# [14 11  1] 421

# indx = 0
# for i in range(21):
#     xyz = meta.colToCoord[indx]
#
#     print(indx,xyz)
#
#     indx = meta.coordToCol[xyz[0]][xyz[1]][xyz[2]]

# ext = input()
#
# exit()

# print(study)
# print(subject)
# print(ntrials)
# print(nvoxels)
# print(dimx)
# print(dimy)
# print(dimz)
# print(colToCoord)
# print(len(colToCoord))
# print(coordToCol)
# print(len(coordToCol))
# print(len(coordToCol[0]))
# print(len(coordToCol[0][0]))

# print(colToCoord[0:999,0])
# print(colToCoord[0:999,1])
# print(colToCoord[0:999,2])
# pl.Plot(colToCoord[0:9999,0],colToCoord[0:9999,1],colToCoord[0:9999,2])

# Plot(colToCoord[0:99,0],colToCoord[0:99,1],colToCoord[0:99,2])


# print(type(mat))

# for x in mat:
#   print(x,'-----------------------------------',mat[x])
#   print('.............')

# print(len(mat))
# print(mat)

# for d in dataArr[:10]:
#
#     dlist = ''
#     for dd in d[:5]:
#         dlist += str(dd) + '           '
#     # print(dlist)

# for trial in info:
#     print(trial)
# print(trial[0],' ',trial[1],' ',trial[2],' ',trial[3],' ',trial[4])

# print(i[:,0:2],' ',i[:,2:4])
# i[:,0:1]
# i[:,1:2]
# x = i[:, 0],i[:, 2]
# condition = dict(zip(condNum, cond))
# c = Counter(condition_word)
# print(len(c.keys()))
# print(c.keys())
# print(c.values())
# print(type(cond))
# print((cond))
# print(type(condNumber))
# print((condNumber))
# for key,value in self.condition_word.items():
#     print(key,' ',value)
# print(condition[10])
# print(self.condition_word[(10,1)])
