import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# Define features and target variable
features = ['WindSpeed', 'Pressure', 'Temperature', 'ZipCode']
target = 'HurricaneHit'

# Create an instance of the HurricaneZipCodePredictor class
predictor = HurricaneZipCodePredictor(features, target)

# Run the training and evaluation process
predictor.run('hurricane_zip_data.csv')

# Example prediction for a specific zip code (e.g., zip code 33101 in Florida)
input_data = pd.DataFrame([[150, 980, 28, 33101]], columns=features)  # Sample input data
likelihood = predictor.predict(input_data)
print(f'The likelihood of a hurricane hitting the zip code 33101 is: {likelihood[0] * 100:.2f}%')
