
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("height_weight_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    height = float(request.form["height"])

    prediction = model.predict([[height]])

    return render_template(
        "index.html",
        prediction=round(prediction[0], 2)
    )

if __name__ == "__main__":
    app.run(debug=True)
