""" Основная программа, вызывающая функцию обработки данных. """

import click
import string
import re
from nltk import word_tokenize
from nltk.corpus import stopwords
import nltk
import pandas as pd


def preprocessing(text):
    """
    Производит предварительную обработку текста.

    Аргументы:
    - text (str): Входной текст для обработки.

    Возвращает:
    - str: Текст после предварительной обработки.
    """

    text = text.lower()
    spec_chars = string.punctuation + "«»\t—…’"
    text = "".join([ch for ch in text if ch not in spec_chars])
    text = re.sub("\n", " ", text)
    text = "".join([ch for ch in text if ch not in string.digits])
    nltk.download("punkt")
    nltk.download("stopwords")
    english_stopwords = stopwords.words("english")
    text_tokens = word_tokenize(text)
    text_tokens = [
        token.strip() for token in text_tokens if token not in english_stopwords
    ]
    return " ".join(text_tokens)


def data_processing(input_filepath, output_filepath):
    """Функция для обработки данных.
    Считывает исходный CSV-файл, выполняет предобработку данных,
    включая удаление пустых значений и дубликатов,
    приведение текста к нижнему регистру,
    и сохраняет обработанный DataFrame в промежуточный CSV-файл."""

    data_frame = pd.read_csv(input_filepath)
    data_frame = data_frame.dropna()
    data_frame = data_frame.drop_duplicates()
    data_frame["comment"] = data_frame["comment"].apply(preprocessing)
    data_frame.to_csv(output_filepath)


@click.command()
@click.option(
    "-i",
    "--input_filepath",
    default="../../data/external/Ethos_Dataset_Binary.csv",
    # type=click.Path(exists=True),
)
@click.option(
    "-o",
    "--output_filepath",
    default="../../data/interim/Ethos_Dataset_Binary_pr.csv",
    # type=click.Path(exists=True),
)
def main(input_filepath, output_filepath):
    data_processing(input_filepath, output_filepath)


if __name__ == "__main__":
    main()
