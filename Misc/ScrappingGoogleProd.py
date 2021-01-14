from time import sleep
from selenium.webdriver import FirefoxProfile, Firefox
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.wait import WebDriverWait
from selenium import webdriver
import chromedriver_binary
from webdriver_manager.chrome import ChromeDriverManager



if __name__ == '__main__':
    # driver = webdriver.Chrome('ChromeDriverManager().install()')
    driver = webdriver.Chrome('/home/az/.wdm/drivers/chromedriver/linux64/86.0.4240.22/chromedriver')
    driver.get('https://stackoverflow.com/users/signup?ssrc=head&returnurl=%2fusers%2fstory%2fcurrent%27')
    sleep(5)
    driver.find_element_by_xpath('//*[@id="openid-buttons"]/button[1]').click()
    driver.find_element_by_xpath('//input[@type="email"]').send_keys('shadowalkerasif@gmail.com')
    driver.find_element_by_xpath('//*[@id="identifierNext"]').click()
    sleep(10)
    driver.find_element_by_xpath('//input[@type="password"]').send_keys('toomuchrepeats333')
    driver.find_element_by_xpath('//*[@id="passwordNext"]').click()
    sleep(5)
    driver.get("https://studio.youtube.com/channel/UCRpNWj49O0yWwH7tIGwXdvQ/videos/upload?filter=%5B%5D&sort=%7B%22columnType%22%3A%22date%22%2C%22sortOrder%22%3A%22DESCENDING%22%7D")
    sleep(5)
    driver.find_element_by_xpath('//*[@id="row-container"]/div[3]/div/div/span').click()
    driver.find_element_by_xpath('//*[@id="privacy-radios"]/div[1]/ytcp-button/div').click()

    # searchTable = WebDriverWait(driver, 10).until(expected_conditions.presence_of_all_elements_located(
    #     (By.XPATH, '//*[@id="chip-bar"]/div')))

    # parentElement = driver.find_element_by_xpath('//*[@id="chip-bar"]/div')
    # elementList = parentElement.find_elements_by_tag_name('ytcp-chip')
    # print(elementList)
    # print(len(elementList))

    s = driver.find_element_by_xpath('//*[@id="text-input"]').text
    print(s)