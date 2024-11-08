import pickle
from flask import Flask, request, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load your model
floodmodel = pickle.load(open('regmodel.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    new_data = np.array(list(data.values())).reshape(1, -1)
    output = floodmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0]) 

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    print("Received input data:", data)  # Debugging print statement
    input_data = np.array(data).reshape(1, -1)
    print("Input array for model:", input_data)  # Debugging print statement
    final_output = floodmodel.predict(input_data)[0]
    print("Prediction output:", final_output)  # Debugging print statement
    return render_template("home.html", prediction_text="The predicted river level is {}".format(final_output))

if __name__ == "__main__":
    app.run(debug=True)




