import re
import essential_generators
import pandas as pd
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize

# corpusFile,wordFreqFile, iteration, genCount = "/home/az/Desktop/corpus.txt","/home/az/Desktop/wordFreq.csv", 3, 10
corpusFile, wordFreqFile, iteration, genCount = "/home/az/Desktop/corpus.txt", "/home/az/Desktop/wordFreq.csv", 300, 1000


def Tokenize(df): return df.apply(lambda row: word_tokenize(row['sentences']), axis=1)


def ProcessParagraph(x):
    df = pd.DataFrame(sent_tokenize(x[0]), columns=['sentences'])
    return df[Tokenize(df).str.len() > 10]


def GenerateandMakeSentences():
    docGen = essential_generators.DocumentGenerator()
    docGen.set_template({'about': 'paragraph'})
    for i in range(iteration):
        with open(corpusFile, "a") as myfile: myfile.write('\n'.join(
            pd.concat(pd.DataFrame(docGen.documents(genCount)).apply(ProcessParagraph, axis=1).values)[
                'sentences']) + '\n')


def WordProbablity():
    with open(corpusFile) as myfile: text = myfile.read()
    allWords = Tokenize(
        pd.DataFrame(re.sub('[^a-zA-Z ]+', '', text).strip().lower().split('\n'), columns=['sentences'])).sum()
    WordStats = pd.DataFrame.from_dict((Counter(allWords)), orient='index', columns=['Count']).sort_values('Count',
                                                                                                           ascending=False)
    (WordStats['Count'] / len(allWords)).round(9).to_csv(wordFreqFile, sep=' ')

    print(len(allWords))
    print(WordStats['Count'].values[-1])
    # print(WordStats)


if __name__ == '__main__':
    # GenerateandMakeSentences()
    WordProbablity()
    exit()
