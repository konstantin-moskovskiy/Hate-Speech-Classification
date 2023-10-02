HateSpeechClassification
==============================

This project performs the task of binary classification to determine the haight-match (eng). It uses machine learning model optimization tools such as Optuna and Hyperopt, as well as Pandas, Polars, and DVC libraries to track experiments and manage data.

Task
------------
The task is to build a model which is able to classify a message - whether it is angry or not

Data
------------
This [dataset](https://github.com/intelligence-csd-auth-gr/Ethos-Hate-Speech-Dataset/blob/master/ethos/ethos_data/Ethos_Dataset_Binary.csv) contains contains 998 comments in the dataset alongside with a label about hate speech presence or absence. 565 of them do not contain hate speech, while the rest of them, 433, contain.

`comment` - message in English

`isHate` - label from 0 to 1

Project Organization
------------

    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── app             <- Graphical application PyQT5 for checking the operation of the program
    │
    ├── notebooks             <- Notebook with EDA
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment
    │
    ├── requirements-linters.txt      <- The requirements file for github actions 
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   └── train_predict_model.py
    │
    └── scripts            <- .sh scripts for the fast .py scripts running


--------

How to run
------------

- You need to install all dependencies with 
```
pip install -r requirements.txt
```

### CLI

You can run the project with [.sh file](https://github.com/konstantin-moskovskiy/Hate-Speech-Classification/tree/main/scripts/start.sh):

```
scripts/start.sh
```









