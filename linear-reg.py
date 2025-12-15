# (Same code I ran — copy/paste into a notebook)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv('weatherHistory.csv')
cols_needed = ["Formatted Date","Temperature (C)","Apparent Temperature (C)","Humidity",
               "Wind Speed (km/h)","Wind Bearing (degrees)","Visibility (km)","Pressure (millibars)",
               "Summary","Precip Type"]
df = df[cols_needed].dropna()

df['Formatted Date parsed'] = pd.to_datetime(df['Formatted Date'], errors='coerce', utc=True)
df = df.dropna(subset=['Formatted Date parsed'])

dt = df['Formatted Date parsed']
df['hour'] = dt.dt.hour
df['month'] = dt.dt.month
df['dayofyear'] = dt.dt.dayofyear
df['weekday'] = dt.dt.weekday
df['year'] = dt.dt.year
df['is_weekend'] = df['weekday'].isin([5,6]).astype(int)

target = "Temperature (C)"
numeric_features = ["Humidity","Wind Speed (km/h)","Wind Bearing (degrees)","Visibility (km)","Pressure (millibars)",
                    "hour","month","dayofyear","weekday","year","is_weekend"]

top_summaries = df['Summary'].value_counts().nlargest(8).index.tolist()
df['Summary_top'] = df['Summary'].where(df['Summary'].isin(top_summaries), other='OTHER')
categorical_features = ['Summary_top','Precip Type']

X = df[numeric_features + categorical_features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

numeric_transformer = Pipeline([('scaler', StandardScaler())])
categorical_transformer = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

preprocessor_no_poly = ColumnTransformer([('num', numeric_transformer, numeric_features),
                                         ('cat', categorical_transformer, categorical_features)])
pipe_lr = Pipeline([('pre', preprocessor_no_poly), ('model', LinearRegression())])
pipe_lr.fit(X_train, y_train)
y_pred_lr = pipe_lr.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

poly_numeric = Pipeline([('scaler', StandardScaler()), ('poly', PolynomialFeatures(degree=2, include_bias=False))])
preprocessor_poly = ColumnTransformer([('poly_num', poly_numeric, numeric_features),
                                      ('cat', categorical_transformer, categorical_features)])
alphas = np.logspace(-3, 3, 13)
pipe_ridge = Pipeline([('pre', preprocessor_poly), ('model', RidgeCV(alphas=alphas, scoring='r2', cv=5))])
pipe_ridge.fit(X_train, y_train)
y_pred_ridge = pipe_ridge.predict(X_test)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

print("Baseline Linear MSE: {:.4f}, R2: {:.4f}".format(mse_lr, r2_lr))
print("Poly Ridge MSE: {:.4f}, R2: {:.4f}, chosen alpha: {:.6f}".format(mse_ridge, r2_ridge, pipe_ridge.named_steps['model'].alpha_))

# Plot
plt.figure(figsize=(8,8))
plt.scatter(y_test, y_pred_ridge, s=6)
plt.xlabel("Actual Temperature (C)")
plt.ylabel("Predicted Temperature (C)")
plt.title("Polynomial (degree=2) + Ridge: Actual vs Predicted Temperature")
mn = min(y_test.min(), y_pred_ridge.min()); mx = max(y_test.max(), y_pred_ridge.max())
plt.plot([mn,mx],[mn,mx], linestyle='--')
plt.xlim(mn, mx); plt.ylim(mn, mx)
plt.show()
