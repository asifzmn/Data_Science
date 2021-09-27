import requests

if __name__ == '__main__':
    r = requests.get('https://bdapis.herokuapp.com/api/v1.1/districts')
    print(r.text)