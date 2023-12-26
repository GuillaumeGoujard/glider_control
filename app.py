import time
import streamlit as st
import numpy as np
import copy
import matplotlib.pyplot as plt
from dynamics import simulate
from model_definition import GliderConstants, kmh_to_ms, SimulationParameters, States, SimulationContainer, \
    WinchConstants
from modes import Mode, get_initial_conditions_from_mode
from plotting import compute_statistics

# Streamlit app starts here
st.title('Glider Simulation')

# Set up the sidebar with inputs for parameters
with st.sidebar:
    st.sidebar.title("Navigation")
    options = ["Simulation", "Metrics"]
    selection = st.sidebar.radio("Choose a page", options)

    # Mode selection
    mode_options = list(Mode)
    selected_mode_name = st.selectbox("Select Mode", options=mode_options, format_func=lambda mode: mode.name)
    # selected_mode = Mode[selected_mode_name]

    st.sidebar.markdown("# Sidebar Options")
    st.header('Simulation Parameters')
    t_final = st.number_input('Final Time (s)', value=100)
    delta_t = st.number_input('Time Step (s)', value=0.1)
    max_distance = st.number_input('Max Distance (m)', value=1000)

    st.header('Glider Constants')
    m = st.number_input('Mass (kg)', value=300.0)
    g = st.number_input('Gravitational Acceleration (m/s^2)', value=9.81)
    rho = st.number_input('Air Density (kg/m^3)', value=1.3)
    S = st.number_input('Wing Area (m^2)', value=15.0)

    # Set defaults based on selected mode
    default_states, default_winch_constants = get_initial_conditions_from_mode(selected_mode_name)

    # States sliders
    st.header("Initial States")
    x = st.slider('Initial X Position (m)', min_value=0, max_value=2000, value=default_states.x, step=1)
    y = st.slider('Initial Y Position (m)', min_value=0.0, max_value=2000.0, value=default_states.y, step=1.0)
    v = st.slider('Initial Velocity (m/s)', min_value=0.0, max_value=200.0, value=default_states.v, step=1.0)
    r = st.slider('Initial Flight Path Angle (rad)', min_value=0.0, max_value=np.pi, value=default_states.r, step=0.01)
    e = st.slider('Initial incidence Angle (rad)', min_value=0.0, max_value=(20/(360))*(2*np.pi), value=default_states.e, step=0.01)


    # Winch constants sliders
    st.header("Winch Constants")
    distance_from_take_off = st.slider('Distance from Take Off (m)', min_value=0, max_value=2000,
                                       value=default_winch_constants.distance_from_take_off, step=1)
    percent_distance_to_drop = st.slider('Percent Distance to Drop (%)', min_value=0, max_value=100,
                                         value=default_winch_constants.percent_distance_to_drop, step=1)
    min_tension = st.slider('Minimum Tension (N)', min_value=0, max_value=10000,
                            value=default_winch_constants.min_tension, step=1)
    max_tension = st.slider('Maximum Tension (N)', min_value=0, max_value=10000,
                            value=default_winch_constants.max_tension, step=1)
    rate_of_change = st.slider('Rate of Change (N/s)', min_value=0, max_value=1000,
                               value=default_winch_constants.rate_of_change, step=1)


# Create objects based on the input parameters
glider_constants = GliderConstants(m=m, g=g, rho=rho, S=S)
winch_constants = WinchConstants(
    distance_from_take_off=distance_from_take_off,
    percent_distance_to_drop=percent_distance_to_drop,
    min_tension=min_tension,
    max_tension=max_tension,
    rate_of_change=rate_of_change
)
initial_states = States(x=x, y=y, v=v, r=r, e=e)
simulation_parameters = SimulationParameters(t_final=t_final, delta_t=delta_t, max_distance=max_distance)
simulation_container = SimulationContainer(initial_conditions=initial_states)

# Run simulation (this will update the simulation_container with the simulated states)
simulate(glider_constants, winch_constants, simulation_parameters, simulation_container)

metrics = compute_statistics(simulation_container)


# Define the different pages
def page_one():
    st.header("Simulation")

    drop_off_x = winch_constants.percent_distance_to_drop / 100 * winch_constants.distance_from_take_off
    # Simulated data for demonstration
    def get_data_for_time(time_step):
        # Get the state at the selected time step
        state_at_time = simulation_container.get_state(time_step)
        # Plot the history as lighter black
        times = delta_t * np.arange(0, time_step)
        x_values = [simulation_container.get_state(time).x for time in range(0, time_step)]
        y_values = [simulation_container.get_state(time).y for time in range(0, time_step)]
        return state_at_time, times, x_values, y_values

    def plot_frame(state_at_time, times, x_values, y_values):
        fig, ax = plt.subplots()

        ax.plot(x_values, y_values, color='lightgrey', label='History')

        # Scatter plot for the current state as a star
        current_x = state_at_time.x
        current_y = state_at_time.y
        ax.scatter(current_x, current_y, color='blue', marker='*', s=100, label='Current State')
        ax.scatter(winch_constants.distance_from_take_off, 0, color='red', label='Winch Location', zorder=5)
        ax.axvline(drop_off_x, color='grey', linestyle='--', label='Drop-off Point')

        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Altitude (m)')
        ax.set_title('Glider Trajectory')
        ax.legend()

        plt.ylim(-5, winch_constants.distance_from_take_off * 1.1)
        return fig

    def generate_forces_in_cartesian(current_state: States, glider: GliderConstants, winch_model: WinchConstants):
        drag, lift = glider.compute_drag_and_lift(velocity=current_state.v, attack_angle=current_state.e)
        winch_tension = winch_model.compute_tension(time=current_state.t, distance_from_starting_point=current_state.x)
        angle_winch_w_vertical = winch_model.compute_angle(altitude=current_state.y,
                                                           horizontal_position=current_state.x)
        angle_w_speed = np.pi / 2 + current_state.r - angle_winch_w_vertical

        vectorial_drag = drag * np.array([-np.cos(current_state.r), np.sin(current_state.r)])
        vectorial_lift = lift * np.array([-np.sin(current_state.r), np.cos(current_state.r)])
        vectorial_weight = glider.m * glider.g * np.array([0, -1])
        vectorial_winch = winch_tension * np.array([np.sin(angle_winch_w_vertical), -np.cos(angle_winch_w_vertical)])
        sum_of_forces = vectorial_drag + vectorial_lift + vectorial_weight + vectorial_winch

        # Plotting

        fig, ax = plt.subplots()

        def plot_vector(v, color, label):
            ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color=color, label=label)

        plot_vector(vectorial_drag, 'r', 'Drag')
        plot_vector(vectorial_lift, 'g', 'Lift')
        plot_vector(vectorial_weight, 'b', 'Weight')
        plot_vector(vectorial_winch, 'y', 'Winch')
        if current_state.y <= 0.1:
            reaction_force = -copy.copy(sum_of_forces)
            reaction_force[0] = 0.0
            sum_of_forces += reaction_force
            plot_vector(reaction_force, 'cyan', 'Gd Reaction')
        plot_vector(sum_of_forces, 'magenta', 'Resultant')
        all_vectors = np.vstack([vectorial_drag, vectorial_lift, vectorial_weight, vectorial_winch, sum_of_forces])
        buffer = 0.1 * np.max(np.abs(all_vectors))  # buffer around the largest vector
        lims = np.max(np.abs(all_vectors)) + buffer
        ax.set_xlim(-lims, lims)
        ax.set_ylim(-lims, lims)
        # Additional plot settings for aesthetics
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.set_aspect('equal')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title('Force Vectors of Glider')
        plt.xlabel('X (N)')
        plt.ylabel('Y (N)')
        plt.subplots_adjust(right=0.8)  # Adjust this value as needed to fit the legend
        return fig

    # Set up the layout
    plot_container = st.empty()
    animation_button = st.button("Play Animation")

    # Animation logic
    if animation_button:
        for t in np.arange(simulation_container.index[0], simulation_container.index[-1], 10):
            fig = plot_frame(*get_data_for_time(t))
            # Display the plot in the specified container
            plot_container.pyplot(fig)
            # Control the animation speed
            time.sleep(0.01)  # Adjust the sleep time to control the frame rate

    # Slider to select the time step for visualization
    time_step = st.slider('Select Time Step for Visualization', min_value=0,
                          max_value=max(simulation_container.index),
                          step=1, value=max(simulation_container.index))
    fig = plot_frame(*get_data_for_time(time_step))
    st.pyplot(fig)
    fig = generate_forces_in_cartesian(simulation_container.get_state(time_step), glider_constants, winch_constants)
    st.pyplot(fig)


    fig, ax = plt.subplots()
    plt.plot(delta_t*np.array(simulation_container.index),
             [simulation_container.get_state(time).v*3.6 for time in simulation_container.index], label='Speed', color='orange')
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (km/h)')
    plt.title('Glider Speed Over Time')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    st.pyplot(fig)


def page_two():
    st.header("Key metrics")
    for k, m in metrics.items():
        st.metric(k, m)




# Main content based on selection
if selection == "Simulation":
    page_one()
elif selection == "Metrics":
    page_two()