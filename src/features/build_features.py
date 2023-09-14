""" Основная программа, вызывающая функцию обработки данных. """

import pandas as pd


def data_processing():
    """ Функция для обработки данных.
    Считывает исходный CSV-файл, выполняет предобработку данных,
    включая удаление пустых значений и дубликатов, приведение текста к нижнему регистру,
    и сохраняет обработанный DataFrame в промежуточный CSV-файл. """

    data_frame = pd.read_csv("../../data/external/Ethos_Dataset_Binary.csv")
    data_frame = data_frame.dropna()
    data_frame = data_frame.drop_duplicates()
    data_frame["comment"] = data_frame["comment"].str.lower()
    data_frame.to_csv("../../data/interim/Ethos_Dataset_Binary_pr.csv")


if __name__ == '__main__':
    data_processing()
