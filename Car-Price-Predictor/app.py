import os
import pandas as pd
import numpy as np
from flask import Flask,render_template,request
import pickle

app = Flask(__name__)

with open('Linear_regression.pkl', 'rb') as f:
    model = pickle.load(f)

df = pd.read_excel("clean_data.xls")
@app.route('/')

def home():
        # Inside your app.py route (likely the main index '/' route)

    # 1. Get unique companies, years, and fuel types
    companies = sorted(df['company'].unique())
    years = sorted(df['year'].unique(), reverse=True)
    fuel_types = df['fuel_type'].unique()

    #  FIX: Build the company_models dictionary using ONLY the 'name' column
    company_models = {}
    for company in companies:
        # Filter the rows for this company, grab ONLY the 'name' column, and get unique values
        models_list = sorted(df[df['company'] == company]['name'].unique().tolist())
        company_models[company] = models_list

    #  Pass it to your render_template
    return render_template('index.html', 
                        companies=companies, 
                        company_models=company_models, # This will now be clean!
                        years=years, 
                        fuel_types=fuel_types,
                        prediction=None,
                        selected_company=None,
                        selected_model=None,
                        selected_year=None,
                        selected_fuel_type=None,
                        selected_kms_driven=None)

@app.route('/predict', methods=['POST'])
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_model')
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel_type')
    kms_driven = int(request.form.get('kilo_driven'))

    # Convert year to car age to match the model's training features
    current_year = 2026
    car_age = current_year - year

    #  Use your loaded model pipeline to make a prediction
    prediction = model.predict(pd.DataFrame(
        [[car_model, company, fuel_type, kms_driven, car_age]],
        columns=['name', 'company', 'fuel_type', 'kms_driven', 'car_age']
    ))

    # Convert the log price back to actual price if the model was trained on log-target
    predicted_price = np.round(np.expm1(prediction[0]), 2)
    prediction_text = f"Estimated Price: Rs {predicted_price}"

    companies = sorted(df['company'].unique())
    years = sorted(df['year'].unique(), reverse=True)
    fuel_types = df['fuel_type'].unique()
    company_models = {}
    for comp in companies:
        company_models[comp] = sorted(df[df['company'] == comp]['name'].unique().tolist())

    return render_template('index.html',
                           companies=companies,
                           company_models=company_models,
                           years=years,
                           fuel_types=fuel_types,
                           prediction=prediction_text,
                           selected_company=company,
                           selected_model=car_model,
                           selected_year=str(year),
                           selected_fuel_type=fuel_type,
                           selected_kms_driven=kms_driven)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
