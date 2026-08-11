import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import pickle


df = pd.read_excel("clean_data.xls")


# Car age generalizes better than a raw year value
CURRENT_YEAR = 2026
df["car_age"] = CURRENT_YEAR - df["year"]
df = df.drop(columns=["year"])

X = df.drop(columns="Price")
y = df["Price"]

# Log-transform the target since car prices are right-skewed.
# train on y_log and convert predictions back with np.expm1() at inference time.
y_log = np.log1p(y)


ohe = OneHotEncoder(handle_unknown="ignore")
ohe.fit(X[["name", "company", "fuel_type"]])

columntransformer = make_column_transformer(
    (OneHotEncoder(categories=ohe.categories_, handle_unknown="ignore"),
     ["name", "company", "fuel_type"]),
    remainder="passthrough"
)


X_train, X_test, y_train, y_test = train_test_split(
    X, y_log, random_state=7, test_size=0.2
)

candidates = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
}

print("Cross-validated R2 scores (5-fold, log-target):")
cv_results = {}
for name, model in candidates.items():
    pipe = make_pipeline(columntransformer, model)
    scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2")
    cv_results[name] = scores.mean()
    print(f" {name}: {scores.mean():.2f} {scores.std():.2f}")

best_name = max(cv_results, key=cv_results.get)
print(f"\nBest baseline model: {best_name}")


if best_name == "Gradient Boosting":
    base_model = GradientBoostingRegressor(random_state=42)
    param_grid = {
        "gradientboostingregressor__n_estimators": [100, 200, 300, 500],
        "gradientboostingregressor__learning_rate": [0.01, 0.05, 0.1],
        "gradientboostingregressor__max_depth": [3, 4, 5, 6],
        "gradientboostingregressor__subsample": [0.7, 0.8, 1.0],
        "gradientboostingregressor__min_samples_leaf": [1, 3, 5],
    }
elif best_name == "Random Forest":
    base_model = RandomForestRegressor(random_state=42)
    param_grid = {
        "randomforestregressor__n_estimators": [100, 200, 300, 500],
        "randomforestregressor__max_depth": [None, 5, 10, 15, 20],
        "randomforestregressor__min_samples_split": [2, 5, 10],
        "randomforestregressor__min_samples_leaf": [1, 2, 4],
    }
else:
    base_model = LinearRegression()
    param_grid = None

pipe = make_pipeline(columntransformer, base_model)

if param_grid:
    search = RandomizedSearchCV(
        pipe, param_grid, n_iter=30, cv=5, scoring="r2",
        random_state=42, n_jobs=-1
    )
    search.fit(X_train, y_train)
    best_pipe = search.best_estimator_
    print(f"\nBest params: {search.best_params_}")
    print(f"Best CV R2 (log-target): {search.best_score_:.4f}")
else:
    pipe.fit(X_train, y_train)
    best_pipe = pipe


y_pred_log = best_pipe.predict(X_test)
y_pred = np.expm1(y_pred_log) # Converting back to log values from actual values
y_test_actual = np.expm1(y_test)

final_r2 = r2_score(y_test_actual, y_pred)
print(f"\nFinal test R2 (actual price scale): {final_r2:.2f}")


with open(f"Linear_regression.pkl", "wb") as f:
    pickle.dump(best_pipe, f)

# Example prediction (remember: model predicts log price,
# so we convert back with expm1)

sample = pd.DataFrame(
    [["Hyundai Grand i10", "Hyundai", 28000, "Petrol", CURRENT_YEAR - 2014]],
    columns=["name", "company", "kms_driven", "fuel_type", "car_age"]
)
predicted_log_price = best_pipe.predict(sample)
predicted_price = np.expm1(predicted_log_price)[0]
print(f"\nSample prediction: Rs {predicted_price:,.2f}")