# movie-prediction
import pandas as pd
import chardet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt

with open('C:/Users/shiva/OneDrive/Desktop/internship encryptix/MOVIE RATING PREDICTION/IMDb Movies India.csv', 'rb') as f:
    result = chardet.detect(f.read())
charenc = result['encoding']
df = pd.read_csv('C:/Users/shiva/OneDrive/Desktop/internship encryptix/MOVIE RATING PREDICTION/IMDb Movies India.csv', encoding=charenc)

df.dropna(inplace=True)
print(df.columns)
X = df.drop('Rating', axis=1)
y = df['Rating']
categorical_columns = [col for col in X.columns if X[col].dtype == 'object']
le = LabelEncoder()
for col in categorical_columns:
    X[col] = le.fit_transform(X[col])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

new_movie_data = {col: 0 for col in categorical_columns}
new_movie_data['Genre'] = 0
new_movie_data['Director'] = 1
new_movie_data['Actor 1'] = 1
new_movie_df = pd.DataFrame([new_movie_data])

oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
oe.fit(X[categorical_columns])  # Fit the OrdinalEncoder to the training data
new_movie_df[categorical_columns] = oe.transform(new_movie_df[categorical_columns])

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')

director_columns = [col for col in df.columns if 'director' in col]

le = LabelEncoder()
for col in director_columns:
    df[col] = le.fit_transform(df[col])

param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20, 30]}
rf = RandomForestRegressor()
grid_search = GridSearchCV(rf, param_grid, cv=3)
grid_search.fit(X_train, y_train)

print(f'Best Parameters: {grid_search.best_params_}')

best_rf = grid_search.best_estimator_
y_pred_best = best_rf.predict(X_test)

predicted_rating = best_rf.predict(new_movie_df)
print(f'Predicted Rating: {predicted_rating[0]:.2f}')
plt.bar(range(1, 2), predicted_rating)
plt.xlabel('New Movie')
plt.ylabel('Predicted Rating')
plt.title('Predicted Rating for New Movie')
plt.show()
