import os
import pandas as pd

main_path = '/home/asif/Datasets/Privacy Policy Dataset/Usable Privacy/OPP-115_v1_0/OPP-115/'
annotations_path = main_path + 'annotations/'
sanitized_policies_path = main_path + 'sanitized_policies/'

if __name__ == '__main__':
    for file in os.listdir(annotations_path):
        df = pd.read_csv(annotations_path+file,header=None)
        print(df.to_string())
        exit()
