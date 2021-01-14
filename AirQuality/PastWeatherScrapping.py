from time import sleep
from selenium.webdriver import FirefoxProfile, Firefox
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.wait import WebDriverWait
from selenium import webdriver
import chromedriver_binary
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.action_chains import ActionChains
import numpy as np
import pandas as pd
from timeit import default_timer as timer
from sklearn.utils.extmath import cartesian
from AirQuality.DataPreparation import LoadMetadata, LoadSeries
import plotly.express as px
from http_request_randomizer.requests.proxy.requestProxy import RequestProxy
import urllib.request
import socket
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import statsmodels.api as sm
from AirQuality.AQ_Analysis import *
from sklearn.preprocessing import StandardScaler
# Time,Celcius,Celcius,%,Direction,mph,mph,in,in,Category

def ConditionStats(allData):
    condition_group = (allData.groupby('Condition').median()).loc[['Haze','Fog','Rain','T-Storm','Drizzle','Thunder','Light Rain','Light Rain with Thunder','Haze / Windy']]
    fig = go.Figure(data=[
        go.Bar(y=condition_group.Reading, x=condition_group.index, marker_color="grey")
    ])
    fig.show()

    direction_group = (allData.groupby('Wind').median())

    fig = go.Figure(data=[
        go.Bar(y=direction_group.Reading, x=direction_group.index, marker_color="grey")
    ])
    fig.show()

def ModelPreparation(timeSeries,reading):
    factors = ['Temperature', 'Humidity', 'Wind Speed']
    timeSeries = timeSeries[factors]
    timeSeries = (timeSeries.resample('H').mean())

    # print(timeSeries.isnull().any(axis=1))
    # print(timeSeries[timeSeries.isnull().any(axis=1)].index)
    # print(reading.loc[timeSeries[timeSeries.isnull().any(axis=1)].index].isnull().sum())
    # MissingDataHeatmap(timeSeries)

    allData = reading.join(timeSeries)
    # PairDistributionSummary(allData)
    # print(allData.info())
    allData = allData.dropna()

    scaler = StandardScaler()
    scaler.fit(allData)
    timeSeries_model = pd.DataFrame(data=scaler.transform(allData),index=allData.index,columns=allData.columns)

    timeSeries_model = pd.concat([timeSeries_model.reset_index().drop('index',axis=1),pd.get_dummies(timeSeries_model.index.month.astype(str), prefix='month'),pd.get_dummies(timeSeries_model.index.hour.astype(str), prefix='hour')],axis=1)
    print(timeSeries_model.info())

    model = sm.OLS(timeSeries_model['Reading'], timeSeries_model.drop('Reading',axis=1))
    results = model.fit()
    print(results.summary())

def PrepareSeasonData():
    timeRange = pd.date_range('2017-01-01', '2019-12-31')
    timeSeries = pd.concat([pd.read_csv(
        '/home/az/PycharmProjects/Data Science/Misc/Past Weather/' + str(singleDate.date())).dropna(how='all') for
                            singleDate in timeRange])
    timeSeries.Time = pd.to_datetime(timeSeries.Time)
    return timeSeries.set_index("Time")

def FactorAnalysis():
    reading = LoadSeries()[['Tungi']]['2017-01-01': '2019-12-31']
    reading.columns = ['Reading']

    timeSeries = PrepareSeasonData()
    # ModelPreparation(timeSeries,reading)
    # return

    print(timeSeries)

    # reading = reading.fillna(reading.median())
    # timeSeries = timeSeries.fillna(timeSeries.median())

    # allData = reading.join(timeSeries)

    # print(timeSeries.Condition.value_counts())
    # print(allData.corr())



    # for f in factors[:]:
    #     fig = px.scatter(allData, y="Reading", x=f)
    #     fig.update_layout(height = 750,width = 750,font=dict(size=21))
    #     fig.show()


def VectorAnalysis():
    timeSeries = PrepareSeasonData()
    print(timeSeries.sample(15).to_string())
    # factors = ['Temperature', 'Humidity', 'Wind Speed']
    # timeSeries = (timeSeries[factors].resample('H').mean())

    # BoxPlotSeason(timeSeries)
    # BoxPlotHour(timeSeries)
    # print(timeSeries.Wind.value_counts())
    # print(timeSeries.Wind.value_counts().sort_index())
    # print(timeSeries.Condition.value_counts())


if __name__ == '__main__':
    FactorAnalysis()
    # VectorAnalysis()
    exit()

    viewButton = '//*[@id="inner-content"]/div[2]/div[1]/div[1]/div[1]/div/lib-date-selector/div/input'
    tableElem = '//*[@id="inner-content"]/div[2]/div[1]/div[5]/div[1]/div/lib-city-history-observation/div/div[2]/table'

    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    prefs = {"translate_whitelists": {"bn": "en"}, "translate": {"enabled": "true"}}
    options.add_experimental_option("prefs", prefs)
    driver = webdriver.Chrome('/home/az/.wdm/drivers/chromedriver/linux64/86.0.4240.22/chromedriver', options=options)
    timeRange = pd.date_range('2019-04-24', '2019-12-31')

    for singleDate in timeRange:
        print(singleDate)
        driver.get('https://www.wunderground.com/history/daily/bd/dhaka/VGHS/date/' + str(singleDate.date()))
        WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.XPATH, viewButton))).click()

        table = WebDriverWait(driver, 180).until(EC.presence_of_element_located((By.XPATH, tableElem)))
        headers = [header.text for header in table.find_elements_by_xpath(".//tr")[0].find_elements_by_xpath(".//th")]
        bodyInfo = [[cell.text for cell in row.find_elements_by_xpath(".//td")] for row in
                    table.find_elements_by_xpath(".//tr")[1:]]

        df = pd.DataFrame(data=bodyInfo, columns=headers).set_index('Time')
        df.index = pd.to_datetime(str(singleDate.date()) + ' ' + df.index)
        df[df.columns.difference(['Time', 'Wind', 'Condition'])] = df[
            df.columns.difference(['Time', 'Wind', 'Condition'])].apply(lambda x: x.str.split().str[0].astype('float'))
        df[['Temperature', 'Dew Point']] = df[['Temperature', 'Dew Point']].apply(lambda x: ((x - 32) * 5 / 9).round(2))
        # df.agg({'Time': lambda x: pd.to_datetime(str(singleDate.date()) + ' ' + x)})
        df.to_csv('/home/az/PycharmProjects/Data Science/Misc/Past Weather/' + str(singleDate.date()))

    # headers = ['Time', 'Temperature', 'Dew Point', 'Humidity', 'Wind', 'Wind Speed', 'Wind Gust', 'Pressure', 'Precip.',
    #  'Condition']
    # bodyInfo =[['12:00 AM', '68 F', '61 F', '78 %', 'CALM', '0 mph', '0 mph', '29.94 in', '0.0 in', 'Fog'],
    #  ['12:30 AM', '68 F', '61 F', '78 %', 'CALM', '0 mph', '0 mph', '29.94 in', '0.0 in', 'Fog'],
    #  ['1:00 AM', '68 F', '63 F', '83 %', 'CALM', '0 mph', '0 mph', '29.94 in', '0.0 in', 'Fog'],
    #  ['1:30 AM', '66 F', '61 F', '83 %', 'CALM', '0 mph', '0 mph', '29.94 in', '0.0 in', 'Fog'],
    #  ['2:00 AM', '66 F', '61 F', '83 %', 'CALM', '0 mph', '0 mph', '29.91 in', '0.0 in', 'Fog'],
    #  ['2:30 AM', '66 F', '61 F', '83 %', 'CALM', '0 mph', '0 mph', '29.91 in', '0.0 in', 'Fog'],
    #  ['3:00 AM', '66 F', '61 F', '83 %', 'CALM', '0 mph', '0 mph', '29.91 in', '0.0 in', 'Fog'],
    #  ['3:30 AM', '66 F', '61 F', '83 %', 'CALM', '0 mph', '0 mph', '29.91 in', '0.0 in', 'Fog'],
    #  ['4:00 AM', '64 F', '61 F', '88 %', 'CALM', '0 mph', '0 mph', '29.91 in', '0.0 in', 'Fog'],
    #  ['4:30 AM', '64 F', '61 F', '88 %', 'CALM', '0 mph', '0 mph', '29.91 in', '0.0 in', 'Fog'],
    #  ['5:00 AM', '64 F', '61 F', '88 %', 'CALM', '0 mph', '0 mph', '29.91 in', '0.0 in', 'Fog'],
    #  ['5:30 AM', '64 F', '61 F', '88 %', 'CALM', '0 mph', '0 mph', '29.91 in', '0.0 in', 'Fog'],
    #  ['6:00 AM', '63 F', '61 F', '94 %', 'CALM', '0 mph', '0 mph', '29.91 in', '0.0 in', 'Fog'],
    #  ['6:30 AM', '63 F', '61 F', '94 %', 'CALM', '0 mph', '0 mph', '29.94 in', '0.0 in', 'Fog'],
    #  ['7:00 AM', '61 F', '59 F', '94 %', 'CALM', '0 mph', '0 mph', '29.94 in', '0.0 in', 'Fog'],
    #  ['7:30 AM', '61 F', '59 F', '94 %', 'CALM', '0 mph', '0 mph', '29.94 in', '0.0 in', 'Fog'],
    #  ['8:00 AM', '63 F', '61 F', '94 %', 'CALM', '0 mph', '0 mph', '29.97 in', '0.0 in', 'Fog'],
    #  ['8:30 AM', '63 F', '61 F', '94 %', 'CALM', '0 mph', '0 mph', '30.00 in', '0.0 in', 'Fog'],
    #  ['9:00 AM', '63 F', '59 F', '88 %', 'CALM', '0 mph', '0 mph', '30.00 in', '0.0 in', 'Fog'],
    #  ['9:30 AM', '64 F', '61 F', '88 %', 'CALM', '0 mph', '0 mph', '30.03 in', '0.0 in', 'Fog'],
    #  ['10:00 AM', '66 F', '61 F', '83 %', 'CALM', '0 mph', '0 mph', '30.03 in', '0.0 in', 'Fog'],
    #  ['10:30 AM', '68 F', '61 F', '78 %', 'CALM', '0 mph', '0 mph', '30.03 in', '0.0 in', 'Fog'],
    #  ['11:00 AM', '70 F', '61 F', '73 %', 'ENE', '6 mph', '0 mph', '30.00 in', '0.0 in', 'Fog'],
    #  ['11:30 AM', '72 F', '61 F', '69 %', 'CALM', '0 mph', '0 mph', '30.00 in', '0.0 in', 'Haze'],
    #  ['12:00 PM', '73 F', '61 F', '65 %', 'CALM', '0 mph', '0 mph', '29.97 in', '0.0 in', 'Haze'],
    #  ['12:30 PM', '75 F', '61 F', '61 %', 'CALM', '0 mph', '0 mph', '29.97 in', '0.0 in', 'Haze'],
    #  ['1:00 PM', '77 F', '57 F', '50 %', 'NE', '6 mph', '0 mph', '29.94 in', '0.0 in', 'Haze'],
    #  ['1:30 PM', '77 F', '54 F', '44 %', 'N', '6 mph', '0 mph', '29.91 in', '0.0 in', 'Haze'],
    #  ['2:00 PM', '79 F', '54 F', '42 %', 'NNE', '7 mph', '0 mph', '29.91 in', '0.0 in', 'Haze'],
    #  ['2:30 PM', '79 F', '54 F', '42 %', 'NNW', '6 mph', '0 mph', '29.88 in', '0.0 in', 'Haze'],
    #  ['3:00 PM', '79 F', '54 F', '42 %', 'NNE', '5 mph', '0 mph', '29.88 in', '0.0 in', 'Haze'],
    #  ['3:30 PM', '79 F', '54 F', '42 %', 'N', '6 mph', '0 mph', '29.88 in', '0.0 in', 'Haze'],
    #  ['4:00 PM', '77 F', '54 F', '44 %', 'NNW', '6 mph', '0 mph', '29.88 in', '0.0 in', 'Haze'],
    #  ['4:30 PM', '77 F', '54 F', '44 %', 'NNW', '6 mph', '0 mph', '29.88 in', '0.0 in', 'Haze'],
    #  ['5:00 PM', '75 F', '54 F', '47 %', 'CALM', '0 mph', '0 mph', '29.88 in', '0.0 in', 'Haze'],
    #  ['5:30 PM', '75 F', '54 F', '47 %', 'CALM', '0 mph', '0 mph', '29.88 in', '0.0 in', 'Haze'],
    #  ['6:00 PM', '73 F', '54 F', '50 %', 'CALM', '0 mph', '0 mph', '29.88 in', '0.0 in', 'Haze'],
    #  ['6:30 PM', '73 F', '55 F', '53 %', 'CALM', '0 mph', '0 mph', '29.88 in', '0.0 in', 'Haze'],
    #  ['7:00 PM', '72 F', '55 F', '57 %', 'CALM', '0 mph', '0 mph', '29.91 in', '0.0 in', 'Haze'],
    #  ['7:30 PM', '72 F', '55 F', '57 %', 'CALM', '0 mph', '0 mph', '29.91 in', '0.0 in', 'Haze'],
    #  ['8:00 PM', '72 F', '55 F', '57 %', 'CALM', '0 mph', '0 mph', '29.91 in', '0.0 in', 'Haze'],
    #  ['8:30 PM', '70 F', '59 F', '68 %', 'CALM', '0 mph', '0 mph', '29.91 in', '0.0 in', 'Haze'],
    #  ['9:00 PM', '68 F', '61 F', '78 %', 'CALM', '0 mph', '0 mph', '29.91 in', '0.0 in', 'Fog'],
    #  ['9:30 PM', '68 F', '63 F', '83 %', 'CALM', '0 mph', '0 mph', '29.91 in', '0.0 in', 'Fog'],
    #  ['10:00 PM', '66 F', '61 F', '83 %', 'CALM', '0 mph', '0 mph', '29.94 in', '0.0 in', 'Fog'],
    #  ['10:30 PM', '66 F', '61 F', '83 %', 'CALM', '0 mph', '0 mph', '29.94 in', '0.0 in', 'Fog'],
    #  ['11:00 PM', '64 F', '59 F', '83 %', 'CALM', '0 mph', '0 mph', '29.94 in', '0.0 in', 'Fog'],
    #  ['11:30 PM', '64 F', '59 F', '83 %', 'CALM', '0 mph', '0 mph', '29.94 in', '0.0 in', 'Fog']]
