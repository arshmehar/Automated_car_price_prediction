from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model (ensure model.pkl is in the same directory)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = [
        data['price'], data['levy'], data['manufacturer'], data['model'],
        data['production_year'], data['category'], data['leather_interior'],
        data['fuel_type'], data['engine_volume'], data['mileage'],
        data['cylinders'], data['gear_box_type'], data['drive_wheels'],
        data['doors'], data['wheels'], data['color'], data['airbags']
    ]
    
    # Ensure proper input format for the model
    prediction = model.predict([np.array(features)])
    return jsonify({'predicted_price': prediction[0]})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
