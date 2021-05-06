
import pandas as pd
import requests

from AirQuality.DataPreparation import LoadMetadata

if __name__ == '__main__':
    api_key = "886553d6fde7834df3190ac259963285"

    # response = requests.get(url).json()
    # with open('weather.json', 'w') as outfile: json.dump(response, outfile)

    # print(data.keys())

    # print(data['lat'],data['lon'],data['timezone'],data['timezone_offset'])
    # print(data['current'])
    # print(len(data['minutely']))


    metaFrame = LoadMetadata().iloc[:]
    for idx,row in metaFrame.iterrows():
        url = "https://api.openweathermap.org/data/2.5/onecall?lat=%s&lon=%s&appid=%s&units=metric" % (row['Latitude'],row['Longitude'], api_key)
        data = requests.get(url).json()
        cur = pd.DataFrame(data['hourly']).drop('weather', axis=1)  # .iloc[:24]
        print(cur.to_string())

    quit()

    # print(getsizeof(cur))
    # cur.to_csv('weather.csv')

    # print(len(cur),type(cur))
    # for c in cur:
    #     print(datetime.fromtimestamp(c['dt']).strftime("%A, %B %d, %Y %I:%M:%S"))

    exit()