from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# ----------------------------
# LOAD MODEL + ENCODERS
# ----------------------------
model = pickle.load(open("model.pkl", "rb"))

enc = pickle.load(open("encoders.pkl", "rb"))

le_item = enc["item"]
le_location = enc["location"]
le_quality = enc["quality"]
le_season = enc["season"]

# ----------------------------
# SAFE TRANSFORM FUNCTION
# ----------------------------
def safe_transform(encoder, value):
    if value not in encoder.classes_:
        return 0  # fallback for unknown values
    return encoder.transform([value])[0]

# ----------------------------
# HOME
# ----------------------------
@app.route('/')
def home():
    return "ML API Running Successfully"

# ----------------------------
# PREDICT API
# ----------------------------
@app.route('/predict', methods=['POST'])
def predict():

    try:
        data = request.get_json()

        # ----------------------------
        # ENCODING INPUTS SAFELY
        # ----------------------------
        item = safe_transform(le_item, data['itemname'])
        location = safe_transform(le_location, data['location'])
        quality = safe_transform(le_quality, data['quality'])
        season = safe_transform(le_season, data['season'])
        lbs = int(data['lbs'])

        # ----------------------------
        # MODEL INPUT
        # ----------------------------
        X = [[item, location, lbs, quality, season]]

        # ----------------------------
        # PREDICTION
        # ----------------------------
        prediction = model.predict(X)[0]

        return jsonify({
            "price": float(prediction)
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "price": 0.0
        })

# ----------------------------
# RUN SERVER
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)