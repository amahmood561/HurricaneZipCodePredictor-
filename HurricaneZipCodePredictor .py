import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


''''
Explanation
__init__: Initializes the class with features and the target variable. Creates a pipeline with scaling and logistic regression for classification.
_create_pipeline: Builds a machine learning pipeline that includes data scaling and logistic regression.
load_data: Loads data from a CSV file containing historical weather and zip code data.
preprocess: Splits the data into features (X) and the target variable (y).
train_test_split: Divides the dataset into training and testing sets.
train: Fits the logistic regression model on the training data.
predict: Takes input weather data and zip code, returning the likelihood of a hurricane hitting that zip code.
evaluate: Evaluates the model's performance on the test set.
run: Orchestrates the full data processing, training, and evaluation workflow.


'''


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
