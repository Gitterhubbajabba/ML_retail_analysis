# Marketing Attribution Analysis for Online Retail
This repository contains a Python script for marketing attribution analysis using online retail data. The code helps businesses understand which marketing channels most effectively drive customer conversions and revenue.
Overview
This analysis tool loads an online retail dataset and performs a comprehensive marketing attribution analysis using machine learning techniques. It's built with pandas, NumPy, matplotlib, seaborn, and scikit-learn libraries, with a focus on understanding customer conversion patterns across different marketing channels.
Features

# Data Exploration & Visualization

Distribution analysis of key features (Quantity, UnitPrice, Country)
Customer behavior analysis
Time series analysis of transactions, customers, and revenue
Correlation analysis between numerical features


# Customer Aggregation

Creates customer-level metrics including number of purchases, total quantity, average price
Calculates customer revenue and identifies unique products purchased
Defines conversion based on revenue or number of purchases


# Marketing Channel Simulation

Simulates customer exposure to various marketing channels (Email, Social Media, Search Engine, Referral, Direct)
Adjusts channel exposure based on conversion status to simulate realistic marketing data


# Predictive Modeling

Prepares data for machine learning with preprocessing pipelines
Builds and trains a logistic regression model
Generates predictions and probability scores


# Model Evaluation & Visualization

Classification report with precision, recall, and F1-score
Confusion matrix visualization
ROC curve analysis
Precision-Recall curve


# Advanced Attribution Analysis

Analyzes marketing channel combinations and their impact on conversion
Creates visual representation of conversion rates by channel combinations
Calculates incremental effect (lift) of each marketing channel
Extracts and visualizes model coefficients to understand channel importance
Provides odds ratios for easier interpretation of channel effects


# Model Persistence

Saves the trained model for future use
Includes instructions for loading and using the saved model



# Requirements

Python 3.x
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib

# Usage

Ensure you have the required libraries installed
Place your online retail dataset as 'online_retail_II.csv' in the same directory
Run the script to perform the full analysis
Review the generated visualizations and model results
Use the saved model for predictions on new data

# Model Output
The script saves the following outputs:

marketing_attribution_model.pkl: The trained pipeline including preprocessing and model
feature_importance.csv: CSV file with feature importance data
Various visualizations saved as PNG files

# Loading the Model
pythonimport joblib
loaded_model = joblib.load('marketing_attribution_model.pkl')

# Make predictions with:
predictions = loaded_model.predict(new_data)
probabilities = loaded_model.predict_proba(new_data)[:, 1]
