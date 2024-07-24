from ipywidgets import interact, widgets
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the trained model and scaler
model = joblib.load('trained_model.pkl')

# Create input widgets for each feature
feature1_widget = widgets.FloatText(description='Feature 1:')
feature2_widget = widgets.FloatText(description='Feature 2:')
# Add more widgets for other features

# Define a function to preprocess input and make predictions
def predict(feature1, feature2):
    # Preprocess the input data using the scaler
    scaled_features = StandardScaler.transform([[feature1, feature2]])  # Modify for more features if needed

    # Make predictions using the loaded model
    prediction = model.predict(scaled_features)[0]
    print(f'The predicted class is: {prediction}')

# Create an interactive user interface using interact
interact(predict, feature1=feature1_widget, feature2=feature2_widget)