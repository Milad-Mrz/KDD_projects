#Data preprocessing scripts: These files contain code that performs data cleaning, feature engineering, data normalization, and other preprocessing tasks. 
# They often take the raw data files as input and output the cleaned and preprocessed data files that can be used for modeling.

import pandas as pd
import numpy as np
from sklearn.utils import resample

def dfCleaner(file_name):
    # Load the dataset into a pandas DataFrame
    df = pd.read_csv(file_name)

    # 1. Check for missing values in each row
    missing_values = df.isnull().any(axis=1)

    # Filter out rows with missing values
    df = df[~missing_values]

    # 2. Handling duplicate data
    duplicates = df.duplicated()
    df = df[~duplicates]

    # 3. Handling invalid data
    # Check and remove rows with invalid values for specified attributes
    valid_season = [1, 2, 3, 4]
    valid_month = list(range(1, 13))
    valid_hour = list(range(24))
    valid_holiday = [0, 1]
    valid_weekday = list(range(7))
    valid_workingday = [0, 1]
    valid_weathersit = [1, 2, 3, 4]

    df = df[df['season'].isin(valid_season)]
    df = df[df['mnth'].isin(valid_month)]
    df = df[df['hr'].isin(valid_hour)]
    df = df[df['holiday'].isin(valid_holiday)]
    df = df[df['weekday'].isin(valid_weekday)]
    df = df[df['workingday'].isin(valid_workingday)]
    df = df[df['weathersit'].isin(valid_weathersit)]

    # 4. Handling outliers and extreme values
    # Assuming you want to handle outliers for all attributes, you can use z-score or IQR
    z_scores = np.abs((df - df.mean()) / df.std())
    df = df[(z_scores < 3).all(axis=1)]  # Remove rows with values outside 3 standard deviations for any attribute

    # 5. Handling data normalization or scaling
    # Normalize all attributes using Min-Max scaling
    
    for column in df.columns:
        if column != 'cnt':
            df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())

    # 6. Addressing imbalance in the target attribute
    # Assuming you want to address the imbalance in the 'cnt' attribute
    # Separate majority and minority classes
    majority_class = df[df['cnt'] < df['cnt'].mean()]
    minority_class = df[df['cnt'] >= df['cnt'].mean()]

    # Upsample minority class
    minority_upsampled = resample(minority_class, replace=True, n_samples=len(majority_class))

    # Combine majority class with upsampled minority class
    df_balanced = pd.concat([majority_class, minority_upsampled])

    # Shuffle the DataFrame if desired
    df_balanced = df_balanced.sample(frac=1)

    # Now, you can use the balanced dataset for further analysis

    # Additional tasks based on your requirements can be performed here


    file_name2 = 'processed_dataset.csv'
    # Example: Saving the processed DataFrame to a new CSV file
    df_balanced.to_csv(file_name2, index=False)

    return df_balanced