
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("data_science_capstone_project_dataset.csv")

df['travel_datetime'] = pd.to_datetime(df['travel_date'] + ' ' + df['travel_time'], format="%d-%m-%y %H:%M")

ride_df = df.groupby('ride_id').agg({
    'seat_number': 'count',
    'travel_datetime': 'first',
    'travel_from': 'first',
    'car_type': 'first',
    'max_capacity': 'first'
}).reset_index()

ride_df.rename(columns={'seat_number': 'seats_sold'}, inplace=True)


ride_df['day_of_week'] = ride_df['travel_datetime'].dt.dayofweek
ride_df['hour'] = ride_df['travel_datetime'].dt.hour
ride_df['is_weekend'] = ride_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)


ride_df = pd.get_dummies(ride_df, columns=['travel_from', 'car_type'], drop_first=True)

X = ride_df.drop(columns=['ride_id', 'travel_datetime', 'seats_sold'])
y = ride_df['seats_sold']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


lr = LinearRegression()
lr.fit(X_train, y_train)

dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)


def evaluate_model(name, model):
    y_pred = model.predict(X_test)
    print(f"\n{name} Evaluation:")
    print("MAE:", round(mean_absolute_error(y_test, y_pred), 2))
    print("RMSE:", round(np.sqrt(mean_squared_error(y_test, y_pred)), 2))
    print("R2 Score:", round(r2_score(y_test, y_pred), 2))

evaluate_model("Linear Regression", lr)
evaluate_model("Decision Tree", dt)
evaluate_model("Random Forest", rf)


plt.figure(figsize=(10, 5))
sns.histplot(ride_df['seats_sold'], kde=True)
plt.title("Distribution of Seats Sold per Ride")

plt.figure(figsize=(10, 5))
sns.boxplot(x='hour', y='seats_sold', data=ride_df)
plt.title("Seats Sold by Hour of the Day")

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=rf.predict(X_test))
plt.xlabel("Actual Seats Sold")
plt.ylabel("Predicted Seats Sold")
plt.title("Random Forest: Actual vs Predicted")

importances = pd.Series(rf.feature_importances_, index=X.columns)
plt.figure(figsize=(12, 6))
importances.sort_values(ascending=False).plot(kind='bar')
plt.title("Feature Importance - Random Forest")
plt.show()
print("\n✅ Project Execution Completed Successfully!")
