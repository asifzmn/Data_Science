import json
from collections import Counter
import pandas as pd
import numpy as np
import re

if __name__ == '__main__':
    chosenList = ['trt','ksto','ghuma','silli','savage','hang','munia','pera','pagol','burger','patta','chocolate','vulval','na','pic']
    list = []

    with open('/home/az/Desktop/Takeout/Hangouts/Hangouts.json') as file: data =  json.load(file)
    for i in range(10000):
        if 'chat_message' in data['conversations'][-1]['events'][i] and  'segment' in data['conversations'][-1]['events'][i]['chat_message']['message_content']:
            if len(data['conversations'][-1]['events'][i]['chat_message']['message_content']['segment'])>1:
                for text in data['conversations'][-1]['events'][i]['chat_message']['message_content']['segment']:
                    if text['type'] == 'TEXT':
                        list.extend(re.sub(r'[^\w\s]','',text['text'].lower()).split(' '))
                    # s.add(text['type'])
    series = pd.Series(list, name='word').value_counts()
    series = series[series > 3]
    series = np.log(series[chosenList])
    series = series/series.sum()
    print(series)
    print(np.random.choice(series.index, 1000, p=series.values))