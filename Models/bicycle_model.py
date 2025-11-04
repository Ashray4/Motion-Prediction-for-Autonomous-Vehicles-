import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def read_dataset(file_path):
    return pd.read_csv(file_path)


def group_data(Data):
    grouped_data = Data.groupby('trackId')
    separated_data = {}

    for track_id, group in grouped_data:
        longitudinal_velocity = group['lonVelocity'].values
        xCenter = group['xCenter'].values
        yCenter = group['yCenter'].values
        heading = group['heading'].values
        lonAcceleration = group['lonAcceleration'].values

        scaler = MinMaxScaler(feature_range=(-1, 1))
        longitudinal_velocity_normalized = scaler.fit_transform(
            longitudinal_velocity.reshape(-1, 1)).flatten()
        xCenter_normalized = scaler.fit_transform(
            xCenter.reshape(-1, 1)).flatten()
        yCenter_normalized = scaler.fit_transform(
            yCenter.reshape(-1, 1)).flatten()
        heading_normalized = scaler.fit_transform(
            heading.reshape(-1, 1)).flatten()
        lonAcceleration_normalized = scaler.fit_transform(
            lonAcceleration.reshape(-1, 1)).flatten()

        separated_data[track_id] = {
            'lonVelocity': longitudinal_velocity_normalized,
            'xCenter': xCenter_normalized,
            'yCenter': yCenter_normalized,
            'heading': heading_normalized,
            'lonAcceleration': lonAcceleration_normalized,
        }
    return separated_data


def calculate_yaw_rate(heading, time_step=0.1):
    yaw_rate = np.diff(heading) / time_step
    # Assume initial yaw rate is the same as the first calculated one
    return np.concatenate(([yaw_rate[0]], yaw_rate))


def predict_positions_and_velocities_bicycle_model(initial_position, initial_velocity, initial_acceleration, initial_heading, yaw_rate, time_horizon, time_step=0.1):
    positions = []
    velocities = []
    accelerations = []
    headings = []
    x, y = initial_position
    v = initial_velocity
    a = initial_acceleration
    heading = initial_heading

    for t in np.arange(0, time_horizon, time_step):
        # Update velocity based on acceleration
        v += a * time_step

        # Update position based on the bicycle model equations
        x += v * np.cos(heading) * time_step
        y += v * np.sin(heading) * time_step

        # Update heading using the yaw rate
        heading += yaw_rate * time_step

        # Store the updated values
        positions.append((x, y))
        velocities.append(v)
        accelerations.append(a)
        headings.append(heading)

    return np.array(positions), np.array(velocities), np.array(accelerations), np.array(headings)


def sliding_window_prediction_bicycle(actual_positions, actual_velocities, actual_accelerations, actual_headings, yaw_rates, window_size=2.0, time_step=0.1):
    time_horizon = window_size
    num_intervals = int((len(actual_positions) * time_step) // time_horizon)

    all_predicted_positions = []
    all_predicted_velocities = []
    all_predicted_accelerations = []
    all_predicted_headings = []

    for i in range(num_intervals - 1):
        start_idx = int(i * (window_size / 2) / time_step)
        end_idx = start_idx + int(window_size / time_step)

        initial_position = actual_positions[start_idx]
        initial_velocity = actual_velocities[start_idx]
        initial_acceleration = actual_accelerations[start_idx]
        initial_heading = actual_headings[start_idx]
        yaw_rate = yaw_rates[start_idx]

        predicted_positions, predicted_velocities, predicted_accelerations, predicted_headings = predict_positions_and_velocities_bicycle_model(
            initial_position=initial_position,
            initial_velocity=initial_velocity,
            initial_acceleration=initial_acceleration,
            initial_heading=initial_heading,
            yaw_rate=yaw_rate,
            time_horizon=time_horizon,
            time_step=time_step
        )

        for j in range(len(predicted_positions)):
            # Apply a gradual blending factor based on the time step
            alpha = 0.3 * (j / len(predicted_positions))
            predicted_positions[j] = alpha * predicted_positions[j] + \
                (1 - alpha) * actual_positions[start_idx + j]
            predicted_velocities[j] = alpha * predicted_velocities[j] + \
                (1 - alpha) * actual_velocities[start_idx + j]
            predicted_accelerations[j] = alpha * predicted_accelerations[j] + \
                (1 - alpha) * actual_accelerations[start_idx + j]

        all_predicted_positions.extend(
            predicted_positions[:len(predicted_positions)//2])
        all_predicted_velocities.extend(
            predicted_velocities[:len(predicted_velocities)//2])
        all_predicted_accelerations.extend(
            predicted_accelerations[:len(predicted_accelerations)//2])
        all_predicted_headings.extend(
            predicted_headings[:len(predicted_headings)//2])

    return np.array(all_predicted_positions), np.array(all_predicted_velocities), np.array(all_predicted_accelerations), np.array(all_predicted_headings)



def plot_trajectories_comparison(actual_positions, predicted_positions, actual_velocities, predicted_velocities, actual_accelerations, predicted_accelerations, actual_headings, predicted_headings):
    time_steps = np.arange(0, len(actual_positions))

    # Calculate Mean Errors
    mean_error_position_x = np.mean(
        np.abs(actual_positions[:, 0] - predicted_positions[:, 0]))
    mean_error_position_y = np.mean(
        np.abs(actual_positions[:, 1] - predicted_positions[:, 1]))
    mean_error_velocity = np.mean(
        np.abs(actual_velocities - predicted_velocities))
    mean_error_acceleration = np.mean(
        np.abs(actual_accelerations - predicted_accelerations))
    mean_error_heading = np.mean(np.abs(actual_headings - predicted_headings))

    plt.figure(figsize=(14, 12))

    # Plot X and Y positions
    plt.subplot(3, 2, 1)
    plt.plot(time_steps, actual_positions[:len(
        time_steps), 0],
             'b-', label='Actual X Position')
    plt.plot(
        time_steps, predicted_positions[:, 0], 'r--', label='Predicted X Position')
    plt.title(
        f'X Position Comparison (Mean Error: {mean_error_position_x:.3f})')
    plt.xlabel('Time Step')
    plt.ylabel('X Position')
    plt.ylim(-4, 4)  # Limit y-axis range
    plt.legend()

    plt.subplot(3, 2, 2)
    plt.plot(time_steps, actual_positions[:, 1],
             'b-', label='Actual Y Position')
    plt.plot(
        time_steps, predicted_positions[:, 1], 'r--', label='Predicted Y Position')
    plt.title(
        f'Y Position Comparison (Mean Error: {mean_error_position_y:.3f})')
    plt.xlabel('Time Step')
    plt.ylabel('Y Position')
    plt.ylim(-4, 4)  # Limit y-axis range
    plt.legend()

    # Plot Longitudinal Velocities
    plt.subplot(3, 2, 3)
    plt.plot(time_steps, actual_velocities, 'b-',
             label='Actual Longitudinal Velocity')
    plt.plot(time_steps, predicted_velocities, 'r--',
             label='Predicted Longitudinal Velocity')
    plt.title(
        f'Longitudinal Velocity Comparison (Mean Error: {mean_error_velocity:.3f})')
    plt.xlabel('Time Step')
    plt.ylabel('Velocity')
    plt.ylim(-4, 4)  # Limit y-axis range
    plt.legend()

    # Plot Accelerations
    plt.subplot(3, 2, 4)
    plt.plot(time_steps, actual_accelerations, 'b-',
             label='Actual Longitudinal Acceleration')
    plt.plot(time_steps, predicted_accelerations, 'r--',
             label='Predicted Longitudinal Acceleration')
    plt.title(
        f'Acceleration Comparison (Mean Error: {mean_error_acceleration:.3f})')
    plt.xlabel('Time Step')
    plt.ylabel('Acceleration')
    plt.ylim(-4, 4)  # Limit y-axis range
    plt.legend()

    # Plot Headings
    plt.subplot(3, 2, 5)
    plt.plot(time_steps, actual_headings, 'b-', label='Actual Heading')
    plt.plot(time_steps, predicted_headings, 'r--', label='Predicted Heading')
    plt.title(f'Heading Comparison (Mean Error: {mean_error_heading:.3f})')
    plt.xlabel('Time Step')
    plt.ylabel('Heading')
    plt.ylim(-4, 4)  # Limit y-axis range
    plt.legend()

    # Plot Trajectory (X vs Y)
    plt.subplot(3, 2, 6)
    plt.plot(actual_positions[:, 0], actual_positions[:,
             1], 'b-', label='Actual Trajectory')
    plt.plot(predicted_positions[:, 0], predicted_positions[:,
             1], 'r--', label='Predicted Trajectory')
    plt.title('Trajectory Comparison')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.ylim(0, 4)  # Limit y-axis range
    plt.ylim(0, 4)  # Limit y-axis range
    plt.legend()

    plt.tight_layout()
    plt.show()


# Example usage:
Data = read_dataset(
    "C:/Users/skspa/OneDrive/Desktop/Project_Seminar/data/raw/00_tracks.csv")
separated_data = group_data(Data)
track_id_example = list(separated_data.keys())[0]

actual_positions = np.vstack((
    separated_data[track_id_example]['xCenter'],
    separated_data[track_id_example]['yCenter']
)).T

# Only considering longitudinal velocity
actual_velocities = separated_data[track_id_example]['lonVelocity']

# Only considering longitudinal acceleration
actual_accelerations = separated_data[track_id_example]['lonAcceleration']

actual_headings = separated_data[track_id_example]['heading']

# Calculate yaw rates
yaw_rates = calculate_yaw_rate(actual_headings)

# Predict positions, velocities, accelerations, and headings using the bicycle model with a sliding window approach
time_step = 0.1
predicted_positions, predicted_velocities, predicted_accelerations, predicted_headings = sliding_window_prediction_bicycle(
    actual_positions,
    actual_velocities,
    actual_accelerations,
    actual_headings,
    yaw_rates,
    window_size=4.0,
    time_step=time_step
)

# Plot the results with comparisons and mean errors
plot_trajectories_comparison(
    actual_positions[:len(predicted_positions)],
    predicted_positions,
    actual_velocities[:len(predicted_velocities)],
    predicted_velocities,
    actual_accelerations[:len(predicted_accelerations)],
    predicted_accelerations,
    actual_headings[:len(predicted_headings)],
    predicted_headings
)
