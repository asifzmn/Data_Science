import os
import shutil
import time
from datetime import date, timedelta, datetime
from distutils.dir_util import copy_tree
from os import listdir
from pathlib import Path
from timeit import default_timer as timer
import pandas as pd
from selenium.webdriver import FirefoxProfile, Firefox
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.wait import WebDriverWait
from time import sleep

from AirQuality.DataPreparation import LoadMetadata


def Scrap():
    list = ['Temperature [1000 hPa]', 'Temperature [850 hPa]', 'Temperature [700 hPa]', 'Wind speed [80 m]',
            'Wind gusts [10 m]', 'Wind speed and direction [900 hPa]', 'Wind speed and direction [850 hPa]',
            'Wind speed and direction [700 hPa]', 'Wind speed and direction [500 hPa]',
            'Sunshine duration (minutes)', 'Solar radiation', 'Direct radiation', 'Diffuse radiation',
            'Precipitation amount', 'Low, mid and high cloud cover', 'Pressure [mean sea level]',
            'Surface skin temperature', 'Soil temperature [0-10 cm down]', 'Soil moisture [0-10 cm down]']

    lastDate, oneDay = datetime.strptime(listdir(savePath)[-1].split(' to ')[1], '%Y-%m-%d').date(), timedelta(days=1)
    datePoints = str(lastDate + oneDay) + ' to ' + str(date.today() - oneDay)
    targetPath = savePath + datePoints
    # targetPath = savePath + '1datePoints'
    if not os.path.exists(targetPath): os.makedirs(targetPath)

    profile = FirefoxProfile()
    profile.set_preference('browser.download.folderList', 2)
    profile.set_preference('browser.download.manager.showWhenStarting', False)
    # profile.set_preference('browser.download.dir', os.getcwd())
    profile.set_preference('browser.download.dir', targetPath)
    # profile.set_preference('browser.helperApps.neverAsk.saveToDisk', 'text/plain')
    profile.set_preference("browser.helperApps.neverAsk.saveToDisk", "text/csv")
    profile.set_preference('general.warnOnAboutConfig', False)
    profile.update_preferences()
    gecko_path = "/media/az/Study/Work/Firefox Web Driver/geckodriver.exe"
    # path = "/media/az/Study/FFWD/Mozilla Firefox/firefox.exe"
    # binary = FirefoxBinary(path)
    driver = Firefox(firefox_profile=profile, executable_path=gecko_path)
    driver.get("https://www.meteoblue.com/en/weather/archive/export/shahbag_bangladesh_7697915")

    driver.find_element_by_id("gdpr_form").click()

    start = timer()
    for index, row in metaFrame.iterrows():
        print(row)
        location = ' '.join(map(str, row[['Latitude', 'Longitude']].values))
        driver.find_element_by_id("gls").send_keys(location + Keys.RETURN)

        searchTable = WebDriverWait(driver, 30).until(expected_conditions.presence_of_all_elements_located(
            (By.XPATH, "//table[@class = 'search_results']//tr")))
        searchTable[1].find_elements_by_xpath(".//td")[1].click()

        factorInput = '//*[@id="wrapper-main"]/div/main/div/div[2]/form/div[5]/div[1]/span/span[1]/span/ul/li[4]/input'
        factors = WebDriverWait(driver, 30).until(expected_conditions.presence_of_all_elements_located
                                                  ((By.XPATH, factorInput)))[0]
        # factors.send_keys('Total cloud cover' + '\n' )
        factors.send_keys('\n'.join(map(str, list)) + '\n')
        sleep(1)

        # for tickBox in ["relhum2m", "pressure", "clouds", "sunshine", "swrad", "directrad", "diffuserad", "windgust",
        #                 "wind+dir80m", "wind+dir900mb", "wind+dir850mb", "wind+dir700mb", "wind+dir500mb", "temp1000mb",
        #                 "temp850mb", "temp700mb", "tempsfc", "soiltemp0to10", "soilmoist0to10"
        #                 ]: driver.find_element_by_id(tickBox).click()

        datePicker = driver.find_element_by_id('daterange')
        datePicker.clear()
        datePicker.send_keys(datePoints + Keys.RETURN)

        driver.find_element_by_name("submit_csv").click()
        # filename = max([targetPath + "/" + f for f in os.listdir(targetPath)], key=os.path.getctime)
        # shutil.move(filename, os.path.join(targetPath, index + '.csv'))
        print(timer() - start)

    time.sleep(15)
    for file, zone in zip(sorted(Path(targetPath).iterdir(), key=os.path.getmtime), metaFrame.index.values):
        shutil.move(file, os.path.join(targetPath, zone + '.csv'))


def readFile(path, index='Dhaka'):
    meteoInfo = pd.read_csv(os.path.join(path, index + '.csv'), sep=',', skiprows=9)
    meteoInfo = meteoInfo.set_index(pd.to_datetime(meteoInfo['timestamp']))
    meteoInfo.drop(['timestamp'], axis=1, inplace=True)
    meteoInfo = meteoInfo.apply(pd.to_numeric)
    print(meteoInfo.index.values[[0, -1]])
    return meteoInfo


def Update():
    savedata, runningData = readFile(savePath), readFile(runningPath)
    # print((runningData[savedata.index.values[-1]+1:]))
    # print(pd.concat([savedata, runningData], ignore_index=False).drop_duplicates())

    if input() != 'yes': return
    copy_tree(savePath, savePathcp)
    for index, row in metaFrame.iterrows():
        with open(os.path.join(savePath, index + '.csv'), 'a') as save:
            save.write("\n" + '\n'.join(
                map(str, open(os.path.join(runningPath, index + '.csv')).read().split('\n')[10:])))


if __name__ == '__main__':
    metaFrame = LoadMetadata()
    runningPath = '/media/az/Study/Air Analysis/AirQuality Dataset/MeteoblueJuly'
    savePath = '/media/az/Study/Air Analysis/AQ Dataset/Meteoblue Scrapped Data/'
    runningPathcp, savePathcp = runningPath + " (copy)", savePath + " (copy)"

    Scrap()
    exit()
