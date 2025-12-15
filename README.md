# 🌤️ Ambient Temperature Prediction using Linear Models

Predict ambient temperature from historical hourly weather data using feature-engineered linear models, culminating in a Polynomial Regression (degree 2) with Ridge regularization achieving an \(R^{2}\) of 0.8387 on ~96k records.[web:10][web:19]

---

## 🚀 Project Snapshot

- **Goal:** Predict ambient temperature (°C) from historical weather data using regression.
- **Best Model:** Polynomial Regression (degree 2) + Ridge Regularization.
- **Final Performance:** \(R^{2} = 0.8387\), MSE ≈ 14.70.
- **Dataset Size:** ~95,936 hourly records from `weatherHistory.csv`.
- **Core Idea:** Push linear regression as far as possible using feature engineering and regularization instead of complex black-box models.[web:10][web:16]

---

## 📌 Problem Statement

Weather prediction depends on complex, non-linear interactions between atmospheric and temporal variables.[web:19]  
This project investigates how far linear regression-based methods can go when combined with systematic feature engineering, polynomial expansion, and Ridge regularization, under the constraints:

- Use **regression** (not classification).
- Achieve \(R^{2} \geq 0.8\).
- **Scientifically justify** every modeling decision.

---

## 📂 Dataset Overview

- **File:** `weatherHistory.csv`
- **Source:** Historical hourly weather observations (temperature, humidity, pressure, wind, etc.).[web:10]

**Features Used**

- **Atmospheric:** `Humidity`, `Pressure`, `WindSpeed`, `Visibility`.
- **Temporal:** Extracted `Hour`, `Month`, `DayOfYear`, `Weekday` from the timestamp.
- **Categorical:** `WeatherSummary`, `PrecipitationType` (one-hot encoded).

**Target Variable**

- **Temperature (°C)** — continuous value to be predicted.

---

## 🔍 Data Processing Pipeline

### Step 1: Raw Dataset Inspection

Loaded `weatherHistory.csv` into a pandas DataFrame and inspected `df.head()` to verify column names, data types, and missing values.[web:10]  
Confirmed presence of timestamp, temperature, humidity, pressure, wind speed, visibility, weather summary, and precipitation-related columns suitable for regression.

> 🖼️ **Dataset Preview**  
> <img width="1194" height="849" alt="image" src="https://github.com/user-attachments/assets/f73717ae-409d-4f52-9f08-4679fb2f4c69" />


---

### Step 2: Feature Engineering (Key Differentiator)

Parsed the datetime column to extract `Hour`, `Month`, `DayOfYear`, and `Weekday` to encode seasonality and daily cycles explicitly.[web:19]  
Applied one-hot encoding to categorical fields (`WeatherSummary`, `PrecipitationType`) and retained scaled atmospheric variables to make them compatible with linear models.

Why this matters:

- Temperature is strongly seasonal and time-dependent; without temporal features, linear models often plateau around \(R^{2} \approx 0.6\) on similar weather datasets.[web:19]
- Enriching the feature space allows linear models to approximate more complex non-linear relationships.

> 🖼️ **Feature Engineering Code**  
> <img width="794" height="204" alt="image" src="https://github.com/user-attachments/assets/994aa16a-6cd3-4cf6-8dc8-ea0ea016d0ca" />


---

## 🧠 Models Implemented

### 1️⃣ Baseline Linear Regression

- **Model:** Ordinary Least Squares Linear Regression on engineered numeric + one-hot encoded features.
- **Test Metrics:**
  - MSE: ~34.09
  - R²: ~0.63

Key observation:

- Purely linear assumptions are not sufficient to capture the curvature and interactions in weather–temperature relationships, leading to underfitting and suboptimal \(R^{2}\).[web:13][web:16]

> 🖼️ **Baseline Output**  
> <img width="501" height="23" alt="image" src="https://github.com/user-attachments/assets/906640d9-5908-4c09-8f8d-0b28983a06db" />


---

### 2️⃣ Polynomial Regression + Ridge Regularization (Final Model)

- Added **polynomial features (degree 2)** on selected numeric variables to capture pairwise interactions and quadratic trends.[web:13][web:16]
- Used **Ridge Regression (L2 regularization)** to control coefficient magnitudes and reduce overfitting from the enlarged feature space.[web:7]
- Performed **cross-validation** over multiple `alpha` values to automatically select the optimal regularization strength.

**Final metrics:**

- MSE: ~14.70
- R²: 0.8387
- Chosen alpha: selected via cross-validated grid search on Ridge, balancing bias–variance trade-off.[web:7]

Why it works better:

- Polynomial features approximate non-linear patterns while remaining linear in parameters.
- Ridge regularization stabilizes the solution, especially when polynomial expansion introduces multicollinearity and many correlated features.[web:7][web:16]

> 🖼️ **Final Model Output**  
> <img width="721" height="36" alt="image" src="https://github.com/user-attachments/assets/b68d87f1-48a5-402c-ba2c-8f4aceecc71a" />


---

## 📈 Actual vs Predicted Visualization

Created a scatter plot with **Actual Temperature (°C)** on the X-axis and **Predicted Temperature (°C)** on the Y-axis, including a diagonal reference line \(y = x\).[web:13][web:16]  
Points cluster closely around the diagonal, indicating that the model predictions align well with true values and errors remain relatively small across the temperature range.

How to interpret:

- Each point represents one test sample.
- Distance from the diagonal equals prediction error; tight clustering indicates strong predictive performance consistent with \(R^{2} = 0.8387\).

> 🖼️ **Actual vs Predicted Temperature**  
> <img width="902" height="893" alt="image" src="https://github.com/user-attachments/assets/238315b7-b1ad-4b55-88b4-898de76cabde" />


---

## 📊 Performance Summary

| Model                          | MSE    | R² Score |
|--------------------------------|--------|----------|
| Baseline Linear Regression     | ~34.09 | ~0.63    |
| Polynomial + Ridge Regression  | ~14.70 | 0.8387   |

The final model meets the project constraint of \(R^{2} \geq 0.8\) while maintaining reasonable error and model interpretability.[web:19]  
Improvements are driven by better feature representation and regularization, not by switching to a fundamentally different model family.

---

## 🧩 Key Concepts Demonstrated

- **Linear Regression limitations:** Standard linear models struggle with non-linear weather patterns and seasonal variations.
- **Feature Engineering impact:** Time-based and encoded categorical features significantly enhance model capacity without changing the core algorithm.[web:10]
- **One-Hot Encoding:** Converts categorical weather descriptors into numeric vectors that linear models can process.
- **Polynomial Feature Expansion:** Adds interaction and squared terms to capture curved relationships in the data.[web:13][web:16]
- **Ridge Regularization:** Applies an L2 penalty to reduce overfitting and improve generalization when feature count grows.[web:7]
- **Cross-Validation:** Systematically selects hyperparameters (like `alpha`) based on held-out validation performance.
- **Model Evaluation:** Uses MSE and R² to quantify error magnitude and explained variance on unseen data.

---

## 🏁 Conclusion

This project shows that model performance depends more on how data is represented than on using complex algorithms.[web:16][web:19]  
By carefully engineering temporal and categorical features, augmenting them with polynomial terms, and applying Ridge regularization to control complexity, a relatively simple regression model achieves strong predictive accuracy on real-world hourly weather data.

---

## 👤 Author

**N. Jayadeep**  
CSE – Cybersecurity  

_Last updated: December 2025_
