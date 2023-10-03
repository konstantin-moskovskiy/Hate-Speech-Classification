
import click
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from transformers import BertForSequenceClassification, BertTokenizer


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
    default="../../models/results_BERT.csv",
    # type=click.Path(exists=True),
)
@click.option(
    "-m",
    "--model_filepath",
    default="../../models/final_model_BERT.pkl",
    # type=click.Path(exists=True),
)
def main(input_filepath, result_filepath, model_filepath):
    data_frame = pd.read_csv(input_filepath)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # model = BertModel.from_pretrained("bert-base-uncased")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def preprocess_text(text):
        tokens = tokenizer.encode(text, add_special_tokens=True)
        return tokens

    def prepare_input_data(inputs):
        """Подготовка входных данных для модели BERT."""
        input_ids = []
        attention_masks = []

        for text in inputs:
            encoded_text = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=512,
                padding=True,
                truncation=True,
            )
            input_ids.append(encoded_text["input_ids"])
            attention_masks.append(encoded_text["attention_mask"])

        max_length = max(len(ids) for ids in input_ids)

        input_ids = torch.tensor(
            [ids + [0] * (max_length - len(ids)) for ids in input_ids]
        )
        attention_masks = torch.tensor(
            [mask + [0] * (max_length - len(mask)) for mask in attention_masks]
        )

        dataset = torch.utils.data.TensorDataset(input_ids, attention_masks)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=16, shuffle=True)

        return dataloader

    def train_model(dataloader):
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=2
        )
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(3):
            model.train()
            total_loss = 0

            for batch in dataloader:
                batch = [item.to(device) for item in batch]
                input_ids, attention_masks = batch

                optimizer.zero_grad()

                outputs = model(input_ids, attention_mask=attention_masks)
                logits = outputs.logits
                loss = criterion(
                    logits, input_ids[:, 1]
                )

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            print("Epoch:", epoch + 1, "Loss:", avg_loss)

    # def save_model():
    #     print()
    #     model.save_pretrained(model_filepath)

    X = data_frame["comment"].tolist()
    y = data_frame["isHate"].apply(round).astype(int).tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    train_dataloader = prepare_input_data(X_train)
    train_model(train_dataloader)
    # save_model()

    print("Training complete")


if __name__ == "__main__":
    main()
