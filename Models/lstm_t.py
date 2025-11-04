import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# Function to preprocess and normalize the dataset


def group_data(Data):
    grouped_data = Data.groupby('trackId')
    separated_data = {}

    for track_id, group in grouped_data:
        longitudinal_velocity = group['lonVelocity'].values
        lateral_velocity = group['latVelocity'].values
        xCenter = group['xCenter'].values
        yCenter = group['yCenter'].values
        xAcceleration = group['xAcceleration'].values
        yAcceleration = group['yAcceleration'].values

        # Normalize the data
        scaler = MinMaxScaler()
        longitudinal_velocity_normalized = scaler.fit_transform(
            longitudinal_velocity.reshape(-1, 1)).flatten()
        lateral_velocity_normalized = scaler.fit_transform(
            lateral_velocity.reshape(-1, 1)).flatten()
        xCenter_normalized = scaler.fit_transform(
            xCenter.reshape(-1, 1)).flatten()
        yCenter_normalized = scaler.fit_transform(
            yCenter.reshape(-1, 1)).flatten()
        xAcceleration_normalized = scaler.fit_transform(
            xAcceleration.reshape(-1, 1)).flatten()
        yAcceleration_normalized = scaler.fit_transform(
            yAcceleration.reshape(-1, 1)).flatten()

        separated_data[track_id] = {
            'longitudinal_velocity': longitudinal_velocity_normalized,
            'lateral_velocity': lateral_velocity_normalized,
            'xCenter': xCenter_normalized,
            'yCenter': yCenter_normalized,
            'xAcceleration': xAcceleration_normalized,
            'yAcceleration': yAcceleration_normalized
        }

    return separated_data

# Preprocessing function to read multiple track IDs and prepare training and testing sets


def pad_or_truncate(sequence, desired_length=100):
    if len(sequence) > desired_length:
        return sequence[:desired_length]  # Truncate
    elif len(sequence) < desired_length:
        padding = np.zeros((desired_length - len(sequence),
                           sequence.shape[1]))  # Pad with zeros
        return np.vstack((sequence, padding))  # Pad
    else:
        return sequence  # No changes if the sequence is already the correct length

# Preprocessing function to read multiple track IDs and prepare training and testing sets


def preprocess_and_split(data_path, track_ids, desired_length=100):
    data = pd.read_csv(data_path)
    preprocessed_data = group_data(data)

    X = []
    y = []
    for track_id in track_ids:
        track_data = preprocessed_data[track_id]
        # Input: longitudinal velocity, lateral velocity, x and y accelerations
        X_track = np.column_stack((track_data['longitudinal_velocity'], track_data['lateral_velocity'],
                                   track_data['xAcceleration'], track_data['yAcceleration']))
        y_track = np.column_stack(
            (track_data['xCenter'], track_data['yCenter']))

        # Pad or truncate each track to the desired length
        X_track = pad_or_truncate(X_track, desired_length)
        y_track = pad_or_truncate(y_track, desired_length)

        X.append(X_track)
        y.append(y_track)

    X = np.array(X)
    y = np.array(y)

    # Use the first 27 track IDs for training and the last 3 for testing
    X_train, X_test, y_train, y_test = X[:-3], X[-3:], y[:-3], y[-3:]

    return X_train, X_test, y_train, y_test

# LSTM model definition with dropout for regularization


def create_lstm_model(input_shape):
    model = Sequential()
    # Return sequences for the full sequence prediction
    model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.3))  # Adding dropout to avoid overfitting
    model.add(Dense(64, activation='relu'))
    # Predicting xCenter and yCenter for each time step
    model.add(Dense(2, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model


# Set the data path and track IDs
data_path = "C:/Users/skspa/Downloads/inD-dataset-v1.1/data/00_tracks.csv"
track_ids = range(31)  # Assuming 31 track IDs available

# Preprocess data
X_train, X_test, y_train, y_test = preprocess_and_split(
    data_path, track_ids, desired_length=100)

# Create the LSTM model
input_shape = (X_train.shape[1], X_train.shape[2])
lstm_model = create_lstm_model(input_shape)

# Train the LSTM model
lstm_model.fit(X_train, y_train, epochs=50, batch_size=32,
               validation_data=(X_test, y_test))

# Predict on the test set
y_pred = lstm_model.predict(X_test)

# Calculate and print the mean squared error and mean absolute error
mse_x = mean_squared_error(y_test[:, :, 0], y_pred[:, :, 0])
mse_y = mean_squared_error(y_test[:, :, 1], y_pred[:, :, 1])
mae_x = mean_absolute_error(y_test[:, :, 0], y_pred[:, :, 0])
mae_y = mean_absolute_error(y_test[:, :, 1], y_pred[:, :, 1])

print(f"Mean Squared Error (xCenter): {mse_x}")
print(f"Mean Squared Error (yCenter): {mse_y}")
print(f"Mean Absolute Error (xCenter): {mae_x}")
print(f"Mean Absolute Error (yCenter): {mae_y}")

# Plot predicted vs actual trajectories for the xCenter and yCenter


def plot_trajectories(y_test, y_pred):
    plt.figure(figsize=(10, 8))

    # Plot xCenter
    plt.subplot(2, 1, 1)
    plt.plot(y_test[:, :, 0].flatten(), color='blue', label='Actual xCenter')
    plt.plot(y_pred[:, :, 0].flatten(), color='red', label='Predicted xCenter')
    plt.title('Actual vs Predicted xCenter')
    plt.xlabel('Time Step')
    plt.ylabel('xCenter')
    plt.legend()

    # Plot yCenter
    plt.subplot(2, 1, 2)
    plt.plot(y_test[:, :, 1].flatten(), color='blue', label='Actual yCenter')
    plt.plot(y_pred[:, :, 1].flatten(), color='red', label='Predicted yCenter')
    plt.title('Actual vs Predicted yCenter')
    plt.xlabel('Time Step')
    plt.ylabel('yCenter')
    plt.legend()

    plt.tight_layout()
    plt.show()


# Call the function to plot predicted vs actual trajectories
plot_trajectories(y_test, y_pred)
