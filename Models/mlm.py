import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Define group_data function (from the preProcessing.py)


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

        # Normalize
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

# Preprocess the dataset and split it


def preprocess_and_split(data_path, track_ids):
    data = pd.read_csv(data_path)
    preprocessed_data = group_data(data)

    X = []
    y = []
    for track_id in track_ids:
        track_data = preprocessed_data[track_id]
        X.append(np.column_stack(
            (track_data['longitudinal_velocity'], track_data['lateral_velocity'])))
        y.append(np.column_stack(
            (track_data['xCenter'], track_data['yCenter'])))

    X = np.vstack(X)
    y = np.vstack(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Define the MLP model


def create_mlp_model(input_shape):
    model = Sequential()
    model.add(Dense(128, input_shape=input_shape, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(2, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

# Smoothing function using Savitzky-Golay filter


def smooth_data(data, window_size=51, poly_order=3):
    return savgol_filter(data, window_length=window_size, polyorder=poly_order)


# Set the data path and track IDs
data_path = "C:/Users/skspa/Downloads/inD-dataset-v1.1/data/00_tracks.csv"
track_ids = range(31)

# Preprocess data
X_train, X_test, y_train, y_test = preprocess_and_split(data_path, track_ids)

# Create the MLP model
input_shape = (X_train.shape[1],)
mlp_model = create_mlp_model(input_shape)

# Train the model
mlp_model.fit(X_train, y_train, epochs=10, batch_size=32,
              validation_data=(X_test, y_test))

# Predict on the test set
y_pred = mlp_model.predict(X_test)

# Apply smoothing to actual and predicted trajectories
y_test_smooth_x = smooth_data(y_test[:, 0])
y_test_smooth_y = smooth_data(y_test[:, 1])
y_pred_smooth_x = smooth_data(y_pred[:, 0])
y_pred_smooth_y = smooth_data(y_pred[:, 1])

# Apply smoothing to velocities
X_test_smooth_velocity_0 = smooth_data(X_test[:, 0])  # longitudinal velocity
X_test_smooth_velocity_1 = smooth_data(X_test[:, 1])  # lateral velocity

# Plot actual vs predicted trajectories, velocities, and accelerations

mse_x = mean_squared_error(y_test[:, 0], y_pred[:, 0])
mse_y = mean_squared_error(y_test[:, 1], y_pred[:, 1])
print(f"Mean Squared Error (xCenter): {mse_x}")
print(f"Mean Squared Error (yCenter): {mse_y}")

def plot_trajectories_velocities_accelerations(y_test_smooth_x, y_test_smooth_y, y_pred_smooth_x, y_pred_smooth_y, X_test_smooth_velocity_0, X_test_smooth_velocity_1):
    fig, axs = plt.subplots(2, 1, figsize=(10, 20))

    # Plot xCenter separately
    axs[0].plot(y_test_smooth_x, color='blue', label='Smoothed Actual xCenter')
    axs[0].plot(y_pred_smooth_x, color='red',
                label='Smoothed Predicted xCenter')
    axs[0].set_title('Smoothed Actual vs Predicted xCenter')
    axs[0].set_xlabel('Time Step')
    axs[0].set_ylabel('xCenter')
    axs[0].legend()

    # Plot yCenter separately
    axs[1].plot(y_test_smooth_y, color='blue', label='Smoothed Actual yCenter')
    axs[1].plot(y_pred_smooth_y, color='red',
                label='Smoothed Predicted yCenter')
    axs[1].set_title('Smoothed Actual vs Predicted yCenter')
    axs[1].set_xlabel('Time Step')
    axs[1].set_ylabel('yCenter')
    axs[1].legend()

    plt.tight_layout()
    plt.show()


# Call the plot function to visualize the smoothed trajectories, velocities, and accelerations
plot_trajectories_velocities_accelerations(
    y_test_smooth_x, y_test_smooth_y,
    y_pred_smooth_x, y_pred_smooth_y,
    X_test_smooth_velocity_0, X_test_smooth_velocity_1
)
