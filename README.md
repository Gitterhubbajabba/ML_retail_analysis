# Marketing Attribution Analysis for Online Retail

This repository contains a Python script for multi-touch marketing attribution analysis using online retail data. The code helps businesses understand which marketing channels and combinations most effectively drive customer conversions and revenue.

## Overview

This analysis tool loads an online retail dataset and performs a comprehensive marketing attribution analysis using machine learning techniques. It builds and saves a model that can be updated with new data as needed.

## Features

- **Data Exploration & Visualization**
  - Distribution analysis of key features (Quantity, UnitPrice, Country)
  - Customer behavior analysis
  - Time series analysis of transactions, customers, and revenue
  - Correlation analysis between numerical features

- **Customer Aggregation**
  - Creates customer-level metrics including number of purchases, total quantity, average price
  - Calculates customer revenue and identifies unique products purchased
  - Defines conversion based on revenue or number of purchases

- **Multi-Touch Attribution Analysis**
  - Simulates and analyzes customer exposure across multiple marketing channels
  - Calculates conversion rates for different channel combinations
  - Performs incremental lift analysis to determine each channel's unique contribution
  - Visualizes channel combinations and their relative effectiveness

- **Multi-Media Channel Integration**
  - Analyzes performance across diverse media channels (Email, Social, Search, Referral, Direct)
  - Creates visualizations showing relative performance of different media types
  - Calculates both absolute and relative lift by channel

- **Predictive Modeling**
  - Builds a machine learning pipeline with preprocessing for both numerical and categorical variables
  - Trains a logistic regression model to predict customer conversions
  - Evaluates model performance using multiple metrics
  - Saves the model for production use and future updates

- **Model Maintenance**
  - Includes code for saving the model for reuse and updating
  - Provides framework for updating the model with new data
  - Includes instructions for loading and using the saved model

- **Advanced Attribution Visualization**
  - Visualizes channel combinations and their impact on conversion
  - Creates bar charts showing lift analysis by channel
  - Provides coefficients and odds ratios for easier interpretation
  - Generates feature importance visualizations

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib

## Usage

1. Ensure you have the required libraries installed
2. Place your online retail dataset as 'online_retail_II.csv' in the same directory
3. Run the script to perform the full analysis
4. Review the generated visualizations and model results
5. Use the saved model for predictions on new data
6. Update the model as needed with new data

## Model Output

The script saves the following outputs:
- `marketing_attribution_model.pkl`: The trained pipeline including preprocessing and model
- `feature_importance.csv`: CSV file with feature importance data
- Various visualizations saved as PNG files including channel_combinations.png and channel_lift_analysis.png

## Loading and Updating the Model

```python
import joblib
loaded_model = joblib.load('marketing_attribution_model.pkl')

# Make predictions with:
predictions = loaded_model.predict(new_data)
probabilities = loaded_model.predict_proba(new_data)[:, 1]

# To update the model with new data:
# 1. Prepare your new data in the same format
# 2. Retrain the model or update with partial_fit() if applicable
# 3. Save the updated model
```

## License

[Add your license information here]
