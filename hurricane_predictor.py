import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class HurricaneZipCodePredictor:
    def __init__(self, features, target):
        self.features = features
        self.target = target
        self.model = self._create_pipeline()

    def _create_pipeline(self):
        # Create a pipeline with scaling and logistic regression
        model_pipeline = Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression())])
        return model_pipeline

    def load_data(self, file_path):
        # Load the dataset
        self.data = pd.read_csv(file_path)
        print(self.data.head())

    def preprocess(self):
        # Split the data into features (X) and target (y)
        self.X = self.data[self.features]
        self.y = self.data[self.target]

    def train_test_split(self, test_size=0.2, random_state=42):
        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state)

    def train(self):
        # Fit the model to the training data
        self.model.fit(self.X_train, self.y_train)

    def predict(self, input_data):
        # Predict the probability of a hurricane hitting a zip code
        return self.model.predict_proba(input_data)[:, 1]  # Probability of hitting (1)

    def evaluate(self):
        # Make predictions on the test set
        y_pred = self.model.predict(self.X_test)
        
        # Evaluate the model
        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred)

        print(f'Accuracy: {accuracy:.2f}')
        print('Classification Report:')
        print(report)

    def run(self, file_path):
        self.load_data(file_path)
        self.preprocess()
        self.train_test_split()
        self.train()
        self.evaluate()

if __name__ == "__main__":
    # Set up argument parser for the CLI
    parser = argparse.ArgumentParser(description='Hurricane Zip Code Predictor CLI')
    parser.add_argument('--data', type=str, required=True, help='Path to the hurricane data CSV file')
    parser.add_argument('--wind_speed', type=float, required=True, help='Wind speed for prediction')
    parser.add_argument('--pressure', type=float, required=True, help='Pressure for prediction')
    parser.add_argument('--temperature', type=float, required=True, help='Temperature for prediction')
    parser.add_argument('--zip_code', type=int, required=True, help='Zip code for prediction')

    args = parser.parse_args()

    # Define features and target variable
    features = ['WindSpeed', 'Pressure', 'Temperature', 'ZipCode']
    target = 'HurricaneHit'

    # Create an instance of the HurricaneZipCodePredictor class
    predictor = HurricaneZipCodePredictor(features, target)

    # Run the training and evaluation process
    predictor.run(args.data)

    # Create input data for prediction using the provided arguments
    input_data = pd.DataFrame([[args.wind_speed, args.pressure, args.temperature, args.zip_code]], columns=features)
    
    # Get the prediction
    likelihood = predictor.predict(input_data)
    print(f'The likelihood of a hurricane hitting the zip code {args.zip_code} is: {likelihood[0] * 100:.2f}%')
