import numpy as np
import matplotlib.pyplot as plt


def plot_trajectory_comparisons(time_steps, actual_positions, predicted_positions, actual_velocities, predicted_velocities, actual_accelerations, predicted_accelerations):
    # Adjusted figure size for better spacing
    fig, axs = plt.subplots(4, 2, figsize=(15, 10))
    fig.suptitle(
        'Trajectory, Velocity, and Acceleration Comparisons', fontsize=16)

    # Calculate Mean Errors
    mean_error_position_x = np.mean(
        np.abs(actual_positions[:len(time_steps), 0] - predicted_positions[:, 0]))
    mean_error_position_y = np.mean(
        np.abs(actual_positions[:len(time_steps), 1] - predicted_positions[:, 1]))
    mean_error_velocity_x = np.mean(
        np.abs(actual_velocities[:len(time_steps), 0] - predicted_velocities[:, 0]))
    mean_error_velocity_y = np.mean(
        np.abs(actual_velocities[:len(time_steps), 1] - predicted_velocities[:, 1]))
    mean_error_acceleration_x = np.mean(np.abs(
        actual_accelerations[:len(time_steps), 0] - predicted_accelerations[:, 0]))
    mean_error_acceleration_y = np.mean(np.abs(
        actual_accelerations[:len(time_steps), 1] - predicted_accelerations[:, 1]))

    # 1. X Position vs Predicted X Position
    axs[0, 0].plot(time_steps, actual_positions[:len(
        time_steps), 0], 'b-', label='Actual X Position')
    axs[0, 0].plot(time_steps, predicted_positions[:, 0],
                   'r--', label='Predicted X Position')
    axs[0, 0].set_title(
        f'Mean Error: {mean_error_position_x:.3f}', pad=10)
    axs[0, 0].set_xlabel('Time Step', labelpad=10)
    axs[0, 0].set_ylabel('X Position', labelpad=10)
    axs[0, 0].set_ylim(-4,4)  # Limit y-axis range
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # 2. Y Position vs Predicted Y Position
    axs[0, 1].plot(time_steps, actual_positions[:len(
        time_steps), 1], 'b-', label='Actual Y Position')
    axs[0, 1].plot(time_steps, predicted_positions[:, 1],
                   'r--', label='Predicted Y Position')
    axs[0, 1].set_title(
        f'Mean Error: {mean_error_position_y:.3f}', pad=10)
    axs[0, 1].set_xlabel('Time Step', labelpad=10)
    axs[0, 1].set_ylabel('Y Position', labelpad=10)
    axs[0, 1].set_ylim(-4,4)  # Limit y-axis range
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # 3. Position Trajectory (X vs Y) for Actual and Predicted
    axs[1, 0].plot(actual_positions[:len(time_steps), 0], actual_positions[:len(
        time_steps), 1], 'b-', label='Actual Trajectory')
    axs[1, 0].plot(predicted_positions[:, 0], predicted_positions[:,
                   1], 'r--', label='Predicted Trajectory', linewidth=1)
    axs[1, 0].set_title('Position Trajectory: Actual vs Predicted', pad=10)
    axs[1, 0].set_xlabel('X Position', labelpad=10)
    axs[1, 0].set_ylabel('Y Position', labelpad=10)
    axs[1, 0].set_ylim(0,4)  # Limit y-axis range
    axs[1, 0].set_xlim(0,4)  # Limit y-axis range
    axs[1, 0].legend()  
    axs[1, 0].grid(True)

    # 4. X Velocity vs Predicted X Velocity
    axs[1, 1].plot(time_steps, actual_velocities[:len(
        time_steps), 0], 'b-', label='Actual X Velocity')
    axs[1, 1].plot(time_steps, predicted_velocities[:, 0],
                   'r--', label='Predicted X Velocity')
    axs[1, 1].set_title(
        f'Mean Error: {mean_error_velocity_x:.3f}', pad=10)
    axs[1, 1].set_xlabel('Time Step', labelpad=10)
    axs[1, 1].set_ylabel('X Velocity', labelpad=10)
    axs[1, 1].set_ylim(-4,4)  # Limit y-axis range
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    # 5. Y Velocity vs Predicted Y Velocity
    axs[2, 0].plot(time_steps, actual_velocities[:len(
        time_steps), 1], 'b-', label='Actual Y Velocity')
    axs[2, 0].plot(time_steps, predicted_velocities[:, 1],
                   'r--', label='Predicted Y Velocity')
    axs[2, 0].set_title(
        f'Mean Error: {mean_error_velocity_y:.3f}', pad=10)
    axs[2, 0].set_xlabel('Time Step', labelpad=10)
    axs[2, 0].set_ylabel('Y Velocity', labelpad=10)
    axs[2, 0].set_ylim(-4,4)  # Limit y-axis range
    axs[2, 0].legend()
    axs[2, 0].grid(True)

    # 6. X Acceleration vs Predicted X Acceleration
    axs[2, 1].plot(time_steps, actual_accelerations[:len(
        time_steps), 0], 'b-', label='Actual X Acceleration')
    axs[2, 1].plot(time_steps, predicted_accelerations[:, 0],
                   'r--', label='Predicted X Acceleration')
    axs[2, 1].set_title(
        f'Mean Error: {mean_error_acceleration_x:.3f}', pad=10)
    axs[2, 1].set_xlabel('Time Step', labelpad=10)
    axs[2, 1].set_ylabel('X Acceleration', labelpad=10)
    axs[2, 1].set_ylim(-4,4)  # Limit y-axis range
    axs[2, 1].legend()
    axs[2, 1].grid(True)

    # 7. Y Acceleration vs Predicted Y Acceleration
    axs[3, 0].plot(time_steps, actual_accelerations[:len(
        time_steps), 1], 'b-', label='Actual Y Acceleration')
    axs[3, 0].plot(time_steps, predicted_accelerations[:, 1],
                   'r--', label='Predicted Y Acceleration')
    axs[3, 0].set_title(
        f'Mean Error: {mean_error_acceleration_y:.3f}', pad=10)
    axs[3, 0].set_xlabel('Time Step', labelpad=10)
    axs[3, 0].set_ylabel('Y Acceleration', labelpad=10)
    axs[3, 0].set_ylim(-4,4)  # Limit y-axis range
    axs[3, 0].legend()
    axs[3, 0].grid(True)

    # Remove the last empty plot by deleting it from the figure
    fig.delaxes(axs[3, 1])

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
