import numpy as np
from matplotlib import pyplot as plt

from model_definition import SimulationContainer, WinchConstants

def compute_statistics(container):
    index = np.array(sorted(container._states.keys()))
    times = np.array([container.get_state(time).x for time in index])
    x_values = np.array([container.get_state(time).x for time in index])
    y_values = np.array([container.get_state(time).y for time in index])
    speeds = np.array([container.get_state(time).v for time in index]) * 3.6  # Convert m/s to km/h

    time_to_land = times[y_values <= 0][0] if any(y_values <= 0) else index[-1]
    max_altitude = np.max(y_values)
    time_to_max_altitude = times[np.argmax(y_values)]
    average_speed = round(np.mean(speeds), 1)

    metrics = {"Time to land (s)":time_to_land,
               "Maximum altitude (m)":max_altitude,
               "Time to reach maximum altitude (s)": time_to_max_altitude,
               "Average speed (km/h)": average_speed}

    return metrics

    # # Print statistics
    # print(f"Time to land (s): {time_to_land}")
    # print(f"Maximum altitude (m): {max_altitude}")
    # print(f"Time to reach maximum altitude (s): {time_to_max_altitude}")
    # print(f"Average speed (km/h): {average_speed}")

# Optional: Visualization of the results
def plot_simulation_results(winch_constants: WinchConstants, container: SimulationContainer):
    index = np.array(sorted(container._states.keys()))
    times = np.array([container.get_state(time).x for time in index])
    x_values = np.array([container.get_state(time).x for time in index])
    y_values = np.array([container.get_state(time).y for time in index])
    speeds = np.array([container.get_state(time).v for time in index]) * 3.6  # Convert m/s to km/h

    # Calculate the x position for the axvline (drop off point)
    drop_off_x = winch_constants.percent_distance_to_drop / 100 * winch_constants.distance_from_take_off

    # IEEE style plotting
    # plt.style.use('ieee')

    # Plot flight path
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(x_values, y_values, label='Flight Path')
    ax1.scatter(winch_constants.distance_from_take_off, 0, color='red', label='Winch Location', zorder=5)
    ax1.axvline(drop_off_x, color='grey', linestyle='--', label='Drop-off Point')
    ax1.set_xlabel('Distance (m)')
    ax1.set_ylabel('Altitude (m)')
    ax1.set_title('Glider Flight Simulation')
    ax1.legend()
    ax1.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(times, speeds, label='Speed', color='orange')
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (km/h)')
    plt.title('Glider Speed Over Time')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()