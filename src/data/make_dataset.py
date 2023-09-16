""" Основная программа, вызывающая функцию загрузки данных. """

import click
import pandas as pd


def download_data(output_file):
    """Загрузка данных из датасета Ethos-Hate-Speech-Dataset.
    Функция считывает CSV-файл, разделенный с помощью символа ';'.
    Загруженные данные сохраняются во внешний CSV-файл в папке
    "../../data/external/"."""

    data_frame = pd.read_csv(
        "https://raw.githubusercontent.com/intelligence-csd-auth-gr"
        "/Ethos-Hate-Speech-Dataset/master/ethos"
        "/ethos_data/Ethos_Dataset_Binary.csv",
        sep=";",
    )
    data_frame.to_csv(output_file)


@click.command()
@click.option(
    "-o",
    "--output_filepath",
    default="../../data/external/Ethos_Dataset_Binary.csv",
    type=click.Path(exists=True),
)
def main(output_filepath):
    download_data(output_filepath)


if __name__ == "__main__":
    main()
