# Car Price Predictor

🔗 Live Demo: [http://192.168.100.89:5000]

A machine learning web app that predicts the resale price of used cars based on make, model, year, mileage, and fuel type, trained on real second-hand car listings.

## Overview

Used car prices depend on a mix of categorical factors (brand, model, fuel type) and numerical factors (age, mileage) that don't have a simple linear relationship. This project scrapes/cleans a real dataset of Quikr car listings, engineers the categorical and numerical features properly, and trains a regression model to estimate a fair price for any car configuration. The app itself is a Flask web form with dynamic dropdowns — selecting a company filters the model list to only that company's cars.

## How It Works

1. **Data** — Started with 892 raw car listings scraped from Quikr (`quikr_car.csv`), containing inconsistent formatting, non-numeric values in numeric columns, and outlier prices.
2. **Cleaning** — Removed unparseable rows, standardized the `year`, `kms_driven`, and `Price` columns to proper numeric types, and filtered unrealistic outlier prices, resulting in a clean dataset of 816 listings across 25 car companies.
3. **Feature Engineering** — Converted `year` into `car_age` (a more generalizable feature than a raw year value), and log-transformed the target `Price` since car prices are right-skewed — this stabilizes the model's error across both cheap and expensive cars.
4. **Feature Encoding** — Used `OneHotEncoder` on the categorical columns (`name`, `company`, `fuel_type`) combined with `ColumnTransformer`, wrapped in an sklearn `Pipeline` so preprocessing and prediction happen in one consistent step.
5. **Fair Model Comparison** — Compared Linear Regression, Random Forest, and Gradient Boosting using **5-fold cross-validation** (not a cherry-picked train/test split). Linear Regression performed best and most consistently on this dataset, followed closely by Random Forest — a reasonable outcome given the dataset's moderate size and largely linear price relationships.
6. **Tuning & Evaluation** — Selected the best-performing model from cross-validation, tuned it with `RandomizedSearchCV`, and evaluated on a held-out test set with predictions converted back to real price scale.
7. **Deployment** — Built a Flask web app with a form UI: selecting a car company dynamically updates the available model dropdown (via a company-to-models lookup), and submitting the form returns a predicted price instantly.

## Tech Stack

- **Language:** Python
- **Data Handling:** Pandas, NumPy
- **Modeling:** Scikit-learn (Linear Regression, Random Forest, Gradient Boosting — compared via cross-validation; Pipeline, ColumnTransformer, OneHotEncoder)
- **Visualization (EDA):** Matplotlib, Seaborn
- **App/Deployment:** Flask, HTML/CSS (Jinja templates)

## Project Structure

```
car-price-predictor/
├── model_building.py                    # Data loading, encoding, model training & selection
├── app.py                               # Flask web app with prediction routes
├── templates/                            # HTML templates for the web form
├── static/                               # CSS/JS assets for the web form
├── car_price_model.pkl                   # Trained model pipeline (best model selected via cross-validation)
├── quikr_car.csv                         # Raw scraped car listings
├── clean_data.xls                        # Cleaned dataset used for training
└── requirements.txt                      # Python dependencies
```

## Installation & Usage

```bash
# Clone the repository
git clone https://github.com/<your-username>/car-price-predictor.git
cd car-price-predictor

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```
The app runs locally at `http://127.0.0.1:5000`.

## Results

| Model              | 5-Fold CV R² |
|--------------------|--------------|
| Linear Regression  | **0.74** (best) |
| Random Forest      | 0.70 |
| Gradient Boosting  | 0.69 |

On a held-out test set, R² averages around **0.55** across different data splits (with natural variance due to the moderate dataset size of ~800 listings). This is reported honestly via cross-validation rather than a single cherry-picked split, giving a realistic estimate of expected real-world performance.

## License

This project is licensed under the [MIT License](LICENSE).

## Disclaimer

This project is built for educational and portfolio purposes, using a limited, real-world scraped dataset of used car listings.
