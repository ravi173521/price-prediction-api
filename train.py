import pandas as pd
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pickle

# ----------------------------
# DATA SETS
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
# GENERATE DATASET
# ----------------------------
data_list = []

for item in items:
    for _ in range(8):  # more data = better accuracy
        data_list.append({
            'itemname': item,
            'location': random.choice(locations),
            'quality': random.choice(qualities),
            'season': random.choice(seasons),
            'lbs': random.randint(50, 200),
            'price': random.randint(20, 150)
        })

df = pd.DataFrame(data_list)

# ----------------------------
# LABEL ENCODING
# ----------------------------
le_item = LabelEncoder()
le_location = LabelEncoder()
le_quality = LabelEncoder()
le_season = LabelEncoder()

df['itemname'] = le_item.fit_transform(df['itemname'])
df['location'] = le_location.fit_transform(df['location'])
df['quality'] = le_quality.fit_transform(df['quality'])
df['season'] = le_season.fit_transform(df['season'])

# ----------------------------
# FEATURES & TARGET
# ----------------------------
X = df[['itemname', 'location', 'lbs', 'quality', 'season']]
y = df['price']

# ----------------------------
# MODEL TRAINING
# ----------------------------
model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    max_depth=12
)

model.fit(X, y)

# ----------------------------
# SAVE MODEL
# ----------------------------
pickle.dump(model, open("model.pkl", "wb"))

#  IMPORTANT (DICT FORMAT - REQUIRED FOR BACKEND)
pickle.dump({
    "item": le_item,
    "location": le_location,
    "quality": le_quality,
    "season": le_season
}, open("encoders.pkl", "wb"))

# ----------------------------
# DONE
# ----------------------------
print("✅ Model trained successfully")
print("📊 Total rows:", len(df))