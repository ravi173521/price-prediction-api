from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
le_item, le_location, le_quality, le_season = pickle.load(open("encoders.pkl", "rb"))

@app.route('/')
def home():
    return "ML API Running"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    item = le_item.transform([data['itemname']])[0]
    location = le_location.transform([data['location']])[0]
    quality = le_quality.transform([data['quality']])[0]
    season = le_season.transform([data['season']])[0]
    qty = data['lbs']

    prediction = model.predict([[item, location, qty, quality, season]])

    return jsonify({"predicted_price": float(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)