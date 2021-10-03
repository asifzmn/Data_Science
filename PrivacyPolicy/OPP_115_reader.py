import os
import pandas as pd

main_path = '/home/asif/Datasets/Privacy Policy Dataset/Usable Privacy/OPP-115_v1_0/OPP-115/'
annotations_path = main_path + 'annotations/'
sanitized_policies_path = main_path + 'sanitized_policies/'


def info_counts():
    info_list = []

    for file in os.listdir(annotations_path):
        df = pd.read_csv(annotations_path + file, header=None)
        # print(df.to_string())
        # print(df.shape)
        info = df.apply(lambda x: x.value_counts().shape[0])
        info_list.append(info)
        # exit()

    df_info = pd.concat(info_list, axis=1)
    print(df_info)
    # print(df_info.std(axis=1))


if __name__ == '__main__':
    info_counts()
    exit()

    for file in os.listdir(annotations_path):
        df = pd.read_csv(annotations_path + file, header=None)
        # print(df.to_string())
        # print(df.shape)
