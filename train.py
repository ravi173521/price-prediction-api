import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pickle

data = pd.DataFrame({
    'itemname': ['Potato', 'Potato', 'Carrot', 'Onion'],
    'location': ['Dublin', 'Cork', 'Galway', 'Dublin'],
    'quality': ['High', 'Medium', 'High', 'Low'],
    'season': ['Winter', 'Winter', 'Summer', 'Summer'],
    'lbs': [200, 180, 150, 160],
    'price': [25, 22, 28, 20]
})

le_item = LabelEncoder()
le_location = LabelEncoder()
le_quality = LabelEncoder()
le_season = LabelEncoder()

data['itemname'] = le_item.fit_transform(data['itemname'])
data['location'] = le_location.fit_transform(data['location'])
data['quality'] = le_quality.fit_transform(data['quality'])
data['season'] = le_season.fit_transform(data['season'])

X = data[['itemname', 'location', 'lbs', 'quality', 'season']]
y = data['price']

model = RandomForestRegressor()
model.fit(X, y)

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump([le_item, le_location, le_quality, le_season], open("encoders.pkl", "wb"))

print("Model trained and saved!")