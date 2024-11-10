# model_server.py
from flask import Flask, request, jsonify
import pickle  # or any other library you’re using to load your model
import mlmodel  # directly import since it's in the same directory
import mlmodelhyperparametertuning
import json
from sklearn.preprocessing import MinMaxScaler

# Initialize the MinMaxScaler
scaler = MinMaxScaler()
app = Flask(__name__)

# Load your ML model here (assuming it’s a pickle model; change as needed)
model = mlmodel.load_model()  # Example: if your model has a `load_model` function
hyperparams = mlmodelhyperparametertuning.get_best_hyperparams()  # Example function

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    normalizedMileage = scaler.fit_transform(data['Mileage'])
    Condition_Score = {"Condition_Score ":(normalizedMileage + (2020 - data['Prod. year']))}
    Safety_Score= {"Safety_Score": data['Airbags']/12}
    updatedData = json.loads(data)
    updatedData.update(Condition_Score)
    updatedData.update(Safety_Score)
    # Perform prediction using your model (adjust input/output format as needed)
    prediction = model.predict(data)
    return jsonify({'prediction': prediction.tolist()})  # return prediction in JSON

if __name__ == '__main__':
    app.run(port=5000)