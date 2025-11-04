import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import glob
import os


def read_and_combine_data(directory):
    # Read all *_tracks.csv files
    all_files = glob.glob(os.path.join(directory, "*_tracks.csv"))
    data = []

    for file in all_files:
        df = pd.read_csv(file)
        data.append(df)

    combined_data = pd.concat(data, ignore_index=True)
    return combined_data


def preprocess_data(data):
    # Select relevant columns
    relevant_columns = ['xCenter', 'yCenter', 'xVelocity',
                        'yVelocity', 'xAcceleration', 'yAcceleration', 'heading']
    data = data[relevant_columns]

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_normalized = scaler.fit_transform(data)

    return data_normalized, scaler


def create_sliding_windows(data, window_size=30, prediction_size=10):
    X, y = [], []

    for i in range(len(data) - window_size - prediction_size):
        X.append(data[i:i+window_size])
        # Predicting only xCenter, yCenter
        y.append(data[i+window_size:i+window_size+prediction_size, :2])

    return np.array(X), np.array(y)


# Example usage
# Replace with your data directory path

