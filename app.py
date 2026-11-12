
import pickle
import numpy as np

from flask import Flask, render_template, request


app = Flask(__name__)

# Load trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)


@app.route("/")
def home():

    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Get input values from HTML form
        hours_studied = float(request.form["hours_studied"])

        previous_scores = float(request.form["previous_scores"])

        extracurricular = int(request.form["extracurricular"])

        sleep_hours = float(request.form["sleep_hours"])

        sample_papers = float(request.form["sample_papers"])

        # Prepare input for prediction
        input_data = np.array([[
            hours_studied,
            previous_scores,
            extracurricular,
            sleep_hours,
            sample_papers
        ]])

        # Make prediction
        prediction = model.predict(input_data)[0]

        prediction = round(prediction, 2)

        return render_template(
            "index.html",
            prediction=prediction
        )

    except (ValueError, KeyError) as e:

        return render_template(
            "index.html",
            error="Please enter valid values."
        )


if __name__ == "__main__":

    app.run(debug=True)
