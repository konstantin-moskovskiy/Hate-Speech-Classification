from flask import Flask, render_template, request
from main import main_check
import os

app = Flask(__name__, template_folder=os.getcwd() + "/templates")


@app.route("/check_hate_speech", methods=["GET", "POST"])
def hate():
    if request.method == "POST":
        s = request.form.get("user_text").strip()
        if s:
            res, color, txt = main_check(s)
            return render_template("main.html", res=res, color=color, txt=txt)
    return render_template("main.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
