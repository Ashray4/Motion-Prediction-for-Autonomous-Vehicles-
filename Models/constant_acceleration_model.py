import numpy as np
import matplotlib.pyplot as plt
from plot_trajectory_comparisons import plot_trajectory_comparisons
from preProcessing import group_data
from readDataset import read_dataset

def predict_positions_and_velocities_constant_acceleration(initial_position, initial_velocity, acceleration, time_horizon, time_step=0.1):
    positions = []
    velocities = []
    accelerations = []
    x, y = initial_position
    v_x, v_y = initial_velocity
    a_x, a_y = acceleration

    # Predict step by step
    for t in np.arange(0, time_horizon, time_step):
        # Update position based on constant acceleration equations
        x += v_x * time_step + 0.5 * a_x * time_step**2
        y += v_y * time_step + 0.5 * a_y * time_step**2

        # Update velocity
        v_x += a_x * time_step
        v_y += a_y * time_step

        positions.append((x, y))
        velocities.append((v_x, v_y))
        accelerations.append((a_x, a_y))
    return np.array(positions), np.array(velocities), np.array(accelerations)


def sliding_window_prediction(actual_positions, actual_velocities, actual_accelerations, window_size=2.0, time_step=0.1):
    time_horizon = window_size
    num_intervals = int((len(actual_positions) * time_step) // time_horizon)

    all_predicted_positions = []
    all_predicted_velocities = []
    all_predicted_accelerations = []

    for i in range(num_intervals - 1):
        start_idx = int(i * (window_size / 2) / time_step)
        end_idx = start_idx + int(window_size / time_step)

        # Get initial conditions
        initial_position = actual_positions[start_idx]
        initial_velocity = actual_velocities[start_idx]
        # Use the first acceleration value only
        initial_acceleration = actual_accelerations[start_idx]

        # Predict for the next window_size seconds
        predicted_positions, predicted_velocities, predicted_accelerations = predict_positions_and_velocities_constant_acceleration(
            initial_position=initial_position,
            initial_velocity=initial_velocity,
            acceleration=initial_acceleration,
            time_horizon=time_horizon,
            time_step=time_step
        )

        alpha = 0.3
        # Weighted update using actual data
        for j in range(len(predicted_positions)):
            predicted_positions[j] = alpha * predicted_positions[j] + \
                (1 - alpha) * actual_positions[start_idx + j]
            predicted_velocities[j] = alpha * predicted_velocities[j] + \
                (1 - alpha) * actual_velocities[start_idx + j]

        all_predicted_positions.extend(
            predicted_positions[:len(predicted_positions)//2])
        all_predicted_velocities.extend(
            predicted_velocities[:len(predicted_velocities)//2])
        all_predicted_accelerations.extend(
            predicted_accelerations[:len(predicted_accelerations)//2])

    return np.array(all_predicted_positions), np.array(all_predicted_velocities), np.array(all_predicted_accelerations)


# Example: Assume you have already extracted the actual_positions, actual_velocities, and actual_accelerations
Data = read_dataset(
    "C:/Users/skspa/OneDrive/Desktop/Project_Seminar/data/raw/00_tracks.csv")
separated_data = group_data(Data)
track_id_example = list(separated_data.keys())[0]

# Extract actual positions (x, y), velocities (xVelocity, yVelocity), and accelerations (xAcceleration, yAcceleration)
actual_positions = np.vstack((
    separated_data[track_id_example]['xCenter'],
    separated_data[track_id_example]['yCenter']
)).T

actual_velocities = np.vstack((
    separated_data[track_id_example]['xVelocity'],
    separated_data[track_id_example]['yVelocity']
)).T

actual_accelerations = np.vstack((
    separated_data[track_id_example]['xAcceleration'],
    separated_data[track_id_example]['yAcceleration']
)).T

# Predict positions and velocities using sliding window approach
time_step = 0.1
predicted_positions, predicted_velocities, predicted_accelerations = sliding_window_prediction(
    actual_positions,
    actual_velocities,
    actual_accelerations,
    window_size=2.0,
    time_step=time_step
)

# Generate time steps for plotting
time_steps = np.arange(0, len(predicted_positions) * time_step, time_step)

# Plot the results
plot_trajectory_comparisons(
    time_steps,
    actual_positions[:len(predicted_positions)],
    predicted_positions,
    actual_velocities[:len(predicted_velocities)],
    predicted_velocities,
    actual_accelerations[:len(predicted_accelerations)],
    predicted_accelerations
)
