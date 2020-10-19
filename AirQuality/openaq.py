import json
import requests

if __name__ == '__main__':
    response = requests.get("https://api.openaq.org/beta/averages")
    response_dict = json.loads(response.text)
    print(response_dict)

