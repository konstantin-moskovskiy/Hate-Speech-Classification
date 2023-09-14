import pandas as pd


def download_data():
    df = pd.read_csv('https://raw.githubusercontent.com/intelligence-csd-auth-gr/Ethos-Hate-Speech-Dataset/master/ethos/ethos_data/Ethos_Dataset_Binary.csv', sep=';')
    df.to_csv("../../data/external/Ethos_Dataset_Binary.csv")

if __name__ == '__main__':
    download_data()