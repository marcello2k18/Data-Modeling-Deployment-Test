import numpy as np
import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib

# Create Flask application
app = Flask(__name__)

# Load the model using joblib
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index2.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get input features from the form and convert them to float
    float_features = [float(x) for x in request.form.values() if x]
    if len(float_features) != 5:
        return render_template("index2.html", prediction_result="Unknown")

    features = [np.array(float_features)]

    # Perform prediction using the loaded model
    prediction = model.predict(features)

    # Pass the prediction result to HTML template
    return render_template("index2.html", prediction_result=prediction)

if __name__ == "__main__":
    app.run(debug=True)

