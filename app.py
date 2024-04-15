import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__)
regmodel = pickle.load(open("regmodel.pkl", "rb"))
scalar = pickle.load(open("scaling.pkl", "rb"))


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict_api", methods=["POST"])
def predict_api():
    try:
        data = request.json["data"]
        new_data = scalar.transform(np.array(list(data.values())).reshape(1, -1))
        output = regmodel.predict(new_data)
        return jsonify({"prediction": output[0]})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = [float(x) for x in request.form.values()]
        final_input = scalar.transform(np.array(data).reshape(1, -1))
        output = regmodel.predict(final_input)[0]
        return render_template(
            "home.html",
            prediction_text="The House price prediction is {}".format(output),
        )
    except Exception as e:
        return render_template("home.html", prediction_text=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)
