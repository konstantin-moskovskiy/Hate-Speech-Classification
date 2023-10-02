import pandas as pd
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setGeometry(200, 200, 600, 300)
        self.setWindowTitle("Hate Speech Classification")
        widget = QWidget()
        self.setCentralWidget(widget)

        vbox = QVBoxLayout()

        label = QLabel("<b>DEFINE HATE SPEECH</b>")
        label.setText("DEFINE HATE SPEECH")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setFont(QFont("Arial", 15))
        vbox.addWidget(label)

        hbox = QHBoxLayout()

        self.text_edit = QTextEdit()
        self.text_edit.setFont(QFont("Arial", 12))
        self.text_edit.setPlaceholderText("Insert a phrase or message")
        self.text_edit.placeholderText()
        hbox.addWidget(self.text_edit)

        btn1 = QPushButton("Check")
        btn1.clicked.connect(self.check)
        hbox.addWidget(btn1)

        self.lbl2 = QLabel()
        self.lbl2.setAlignment(Qt.AlignmentFlag.AlignCenter)

        vbox.addLayout(hbox)
        vbox.addWidget(self.lbl2)

        widget.setLayout(vbox)

        self.data_frame = pd.read_csv("../data/interim/Ethos_Dataset_Binary_pr.csv")
        del self.data_frame["Unnamed: 0.1"]
        del self.data_frame["Unnamed: 0"]

        self.show()

    def check(self):
        s = self.text_edit.toPlainText()
        self.lbl2.setFont(QFont("Arial", 15))

        new_row = {"comment": s, "isHate": 0}
        df2 = pd.DataFrame([new_row])
        data_frame = pd.concat([self.data_frame, df2], ignore_index=True)
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(data_frame["comment"])
        user = X[-1]
        with open("../models/final_model.pkl", "rb") as file:
            model = pickle.load(file)
            res = model.predict(user)
            res = str(res.tolist()[0])
            if res == "1":
                self.lbl2.setStyleSheet("color: red; font-weight: bold;")
                self.lbl2.setText("You're being rude! Be polite")
            else:
                self.lbl2.setStyleSheet("color: green; font-weight: bold;")
                self.lbl2.setText("I like your message")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    sys.exit(app.exec_())
