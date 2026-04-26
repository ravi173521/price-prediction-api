import pandas as pd
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pickle

# ----------------------------
# ITEM LISTS
# ----------------------------
fruits = [
    'Apple','Banana','Orange','Mango','Grapes','Pineapple',
    'Watermelon','Papaya','Guava','Pomegranate','Strawberry'
]

vegetables = [
    'Potato','Carrot','Onion','Tomato','Spinach','Cabbage',
    'Cauliflower','Brinjal','Chili','Cucumber','Pumpkin'
]

rices = [
    'Basmati Rice','Sona Masoori','Brown Rice','Jasmine Rice',
    'Idli Rice','Parboiled Rice'
]

items = fruits + vegetables + rices

locations = ['Dublin', 'Cork', 'Galway']
seasons = ['Winter', 'Summer', 'Monsoon', 'Autumn']
qualities = ['High', 'Medium', 'Low']

# ----------------------------
# DATA GENERATION (NO MANUAL REPEAT)
# ----------------------------
data_list = []

for item in items:
    for _ in range(5):  # 5 samples per item
        data_list.append({
            'itemname': item,
            'location': random.choice(locations),
            'quality': random.choice(qualities),
            'season': random.choice(seasons),
            'lbs': random.randint(50, 200),
            'price': random.randint(20, 120)
        })

data = pd.DataFrame(data_list)

# ----------------------------
# LABEL ENCODING
# ----------------------------
le_item = LabelEncoder()
le_location = LabelEncoder()
le_quality = LabelEncoder()
le_season = LabelEncoder()

data['itemname'] = le_item.fit_transform(data['itemname'])
data['location'] = le_location.fit_transform(data['location'])
data['quality'] = le_quality.fit_transform(data['quality'])
data['season'] = le_season.fit_transform(data['season'])

# ----------------------------
# FEATURES & TARGET
# ----------------------------
X = data[['itemname', 'location', 'lbs', 'quality', 'season']]
y = data['price']

# ----------------------------
# MODEL TRAINING
# ----------------------------
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X, y)

# ----------------------------
# SAVE MODEL
# ----------------------------
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump([le_item, le_location, le_quality, le_season], open("encoders.pkl", "wb"))

print("Model trained successfully with fruits, vegetables & rice dataset!")
print("Total rows:", len(data))