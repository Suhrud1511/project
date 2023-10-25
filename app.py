import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the models and scalers for different algorithms
linear_reg_model = pickle.load(open('linear_regression_model.pkl', 'rb'))
xgboost_model = pickle.load(open('xgboost_model.pkl', 'rb'))
linear_reg_scaler = pickle.load(open('scalar.pkl', 'rb'))
xgboost_scaler = pickle.load(open('scalar.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user inputs from the form
        data = [float(x) for x in request.form.values()]
        input_data = np.array(data).reshape(1, -1)

        # Check which model the user has selected
        selected_model = request.form['model']

        # Predict based on the selected model
        if selected_model == 'linear':
            input_data_scaled = linear_reg_scaler.transform(input_data)
            prediction = linear_reg_model.predict(input_data_scaled)[0]
        elif selected_model == 'xgboost':
            input_data_scaled = xgboost_scaler.transform(input_data)
            prediction = xgboost_model.predict(input_data_scaled)[0]
        else:
            return render_template("home.html", prediction_text="Invalid model selection")

        return render_template("home.html", prediction_text=f"The House price prediction is {prediction:.2f}")
    except Exception as e:
        return render_template("home.html", prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
