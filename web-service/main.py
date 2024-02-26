import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def main_check(s: str):
    new_row = {"comment": s, "isHate": 0}
    data_frame = pd.read_csv("main/Ethos_Dataset_Binary_pr.csv")
    del data_frame["Unnamed: 0.1"]
    del data_frame["Unnamed: 0"]
    df2 = pd.DataFrame([new_row])
    data_frame = pd.concat([data_frame, df2], ignore_index=True)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data_frame["comment"])
    user = X[-1]
    try:
        with open("main/final_model.pkl", "rb") as file:
            model = pickle.load(file)
            res = model.predict(user)
            res = str(res.tolist()[0])
            if res == "1":
                return "You're being rude! Be polite", "red", s
            else:
                return "I like your message", "green", s
    except Exception:
        return (
            """I can't classify this message right now.\n
            Let's wait for the update together...""",
            "black",
            s,
        )
