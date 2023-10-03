import click
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


def train_lstm(data):
    def objective(trial):
        num_words = 10000
        embedding_dim = trial.suggest_categorical("embedding_dim",
                                                  [50, 100, 150])
        lstm_units = trial.suggest_categorical("lstm_units",
                                               [64, 128, 256])
        learning_rate = trial.suggest_loguniform("learning_rate",
                                                 1e-5, 1e-1)

        tokenizer = Tokenizer(num_words=num_words, oov_token="OOV")
        tokenizer.fit_on_texts(data["comment"])
        sequences = tokenizer.texts_to_sequences(data["comment"])
        max_sequence_len = max([len(sequence) for sequence in sequences])
        padded_sequences = pad_sequences(
            sequences, maxlen=max_sequence_len, padding="post"
        )

        X_train, X_test, y_train, y_test = train_test_split(
            padded_sequences, data["isHate"].values,
            test_size=0.2, random_state=42)

        model = Sequential()
        model.add(Embedding(num_words, embedding_dim,
                            input_length=max_sequence_len))
        model.add(LSTM(lstm_units))
        model.add(Dense(1, activation="sigmoid"))

        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss="binary_crossentropy", metrics=["accuracy"])
        model.fit(X_train, y_train, epochs=10, batch_size=16)

        y_pred_proba = model.predict(X_test).ravel()
        y_pred = (y_pred_proba > 0.5).astype(int)
        roc_auc = roc_auc_score(y_test, y_pred)
        return roc_auc

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    best_params = study.best_params

    num_words = 10000
    embedding_dim = best_params["embedding_dim"]
    lstm_units = best_params["lstm_units"]
    learning_rate = best_params["learning_rate"]

    tokenizer = Tokenizer(num_words=num_words, oov_token="OOV")
    tokenizer.fit_on_texts(data["comment"])
    sequences = tokenizer.texts_to_sequences(data["comment"])
    max_sequence_len = max([len(sequence) for sequence in sequences])
    padded_sequences = pad_sequences(sequences,
                                     maxlen=max_sequence_len, padding="post")

    X_train, X_test, y_train, y_test = train_test_split(
        padded_sequences, data["isHate"].values, test_size=0.2, random_state=42
    )

    model = Sequential()
    model.add(Embedding(num_words, embedding_dim,
                        input_length=max_sequence_len))
    model.add(LSTM(lstm_units))
    model.add(Dense(1, activation="sigmoid"))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=10, batch_size=16)

    y_pred_proba = model.predict(X_test).ravel()
    y_pred = (y_pred_proba > 0.5).astype(int)
    roc_auc = roc_auc_score(y_test, y_pred)
    return model, roc_auc


@click.command()
@click.option(
    "-i",
    "--input_filepath",
    default="../../data/interim/Ethos_Dataset_Binary_pr.csv",
    # type=click.Path(exists=True),
)
@click.option(
    "-r",
    "--result_filepath",
    default="../../models/results_LSTM.csv",
    # type=click.Path(exists=True),
)
@click.option(
    "-m",
    "--model_filepath",
    default="../../models/final_model_LSTM.pkl",
    # type=click.Path(exists=True),
)
def main(input_filepath, result_filepath, model_filepath):
    preprocessed_data = pd.read_csv(input_filepath)
    lstm_model, roc_auc = train_lstm(preprocessed_data)
    lstm_model.save(model_filepath)
    results_df = pd.DataFrame({"Model": ["LSTM"], "ROC-AUC": [roc_auc]})
    results_df.to_csv(result_filepath, index=False)
    print("Best ROC-AUC score:", roc_auc)


if __name__ == "__main__":
    main()
