from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
model = pickle.load(open('D:/Flask Web Apps/House Price Prediction/Model and Requirement/model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    try:
        # Extract features from form data
        feature = [int(x) for x in request.form.values()]
        feature_final = np.array(feature).reshape(1, -1)  # Ensure correct shape for prediction
        
        # Predict the price
        prediction = model.predict(feature_final)
        
        return render_template('index.html', prediction_text='Price of House will be Rs. {}'.format(int(prediction[0])))
    except Exception as e:
        return render_template('index.html', prediction_text='Error: {}'.format(str(e)))

if __name__ == '__main__':
    app.run(debug=True)
