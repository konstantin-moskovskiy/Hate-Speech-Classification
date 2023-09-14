import pandas as pd

def data_processing():
    df = pd.read_csv("../../data/external/Ethos_Dataset_Binary.csv")
    df = df.dropna()
    df = df.drop_duplicates()
    df["comment"] = df["comment"].str.lower()
    df.to_csv("../../data/interim/Ethos_Dataset_Binary_pr.csv")

if __name__ == '__main__':
    data_processing()