import json
import pandas as pd

if __name__ == '__main__':
    with open('/home/az/Desktop/upazilla.geojson') as file: data = json.load(file)
    # print(data.keys())
    data['features'] = [feature for feature in data['features'] if feature['properties']['NAME_3'] == 'Cox\'s Bazar']
    # df = pd.DataFrame([feature['properties'] for feature in data['features']])
    # print(df.to_string())
    with open('/home/az/Desktop/upazilla_Cox\'s Bazar.geojson', 'w') as outfile: json.dump(data, outfile)
    exit()
