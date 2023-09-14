""" Основная программа, вызывающая функцию загрузки данных. """

import pandas as pd


def download_data():
    """ Функция для загрузки данных из датасета Ethos-Hate-Speech-Dataset.
    Функция считывает CSV-файл, расположенный по ссылке, и разделенный с помощью символа ';'.
    Загруженные данные сохраняются во внешний CSV-файл в папке "../../data/external/"."""

    data_frame = pd.read_csv(
        'https://raw.githubusercontent.com/intelligence-csd-auth-gr'
        '/Ethos-Hate-Speech-Dataset/master/ethos/ethos_data/Ethos_Dataset_Binary.csv',
        sep=';')
    data_frame.to_csv("../../data/external/Ethos_Dataset_Binary.csv")


if __name__ == '__main__':
    download_data()
