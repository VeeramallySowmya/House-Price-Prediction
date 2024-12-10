import uvicorn
from fastapi import FastAPI
from PropertyVariables import PropertyPricePred
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import VotingRegressor
import warnings


warnings.filterwarnings("ignore")


HousePricePredictionApp = FastAPI()

fileName = r'C:\Users\sowmy\OneDrive\Desktop\IS\Housing_Price_Prediction_API\property_price_prediction_voting.sav' 
loaded_model = joblib.load(fileName)

@HousePricePredictionApp.get('/')
def index():
    return {'message': 'Welcome to Price Prediction World!'}

@HousePricePredictionApp.post('/predict')
def predict_price(data: PropertyPricePred):
    data = data.model_dump()
    print(data)
    data = pd.DataFrame([data])
    print(data.head())

    prediction = loaded_model.predict(data)
    print("Raw Prediction:", prediction)

    # Handle negative predictions (if applicable)
    prediction = np.clip(prediction, a_min=0, a_max=None)
    print("Adjusted Prediction:", prediction)
    
    print(f'Prediction: {prediction}')  # For debugging
    return {'predicted_price': prediction[0]}  # Return as JSON
    

if __name__ == '__main__':
    uvicorn.run("app:HousePricePredictionApp",host='127.0.0.1', port=8000, reload=True, workers=3)


