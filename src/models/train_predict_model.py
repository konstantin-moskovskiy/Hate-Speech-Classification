""" Основная программа для обучения модели"""

import pickle
import optuna
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def main():
    """ Основная функция, выполняющая обучение модели,
    сохранение результатов и вывод лучших параметров."""

    data_frame = pd.read_csv("../../data/interim/Ethos_Dataset_Binary_pr.csv")
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data_frame['comment'])
    y = data_frame['isHate'].astype(int)
    lb = LabelBinarizer()
    y = lb.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    def objective(trial):
        """ Целевая функция для оптимизации hyperopt.
        Параметры:
            trial: объект Trial из optuna

        Возвращает:
            accuracy: точность модели """

        C = trial.suggest_loguniform('C', 1e-5, 1e5)
        max_iter = trial.suggest_int('max_iter', 100, 500, 1000)
        solver = trial.suggest_categorical('solver',
                                           ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'])
        model = LogisticRegression(C=C, max_iter=max_iter, solver=solver)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy

    def save_model(study):
        """ Сохраняет обученную модель в файл.

        Параметры:
            study: объект Study из optuna """

        final_model = LogisticRegression(C=study.best_trial.params['C'],
                                         max_iter=study.best_trial.params['max_iter'],
                                         solver=study.best_trial.params['solver'])
        final_model.fit(X_train, y_train)
        filename = '../../models/final_model.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(final_model, file)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    save_model(study)
    results_df = study.trials_dataframe()
    results_df.to_csv('../../models/results.csv', index=False)
    print('Best trial:', study.best_trial.params)
    print('Best accuracy:', study.best_trial.value)


if __name__ == '__main__':
    main()
