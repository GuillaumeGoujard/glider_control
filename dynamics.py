import copy
import logging
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import copy
from model_definition import States, GliderConstants, SimulationParameters, SimulationContainer, WinchConstants

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_forces_in_cartesian(current_state: States, glider: GliderConstants, winch_model: WinchConstants):
    drag, lift = glider.compute_drag_and_lift(velocity=current_state.v, attack_angle=current_state.e)
    winch_tension = winch_model.compute_tension(time=current_state.t, distance_from_starting_point=current_state.x)
    angle_winch_w_vertical = winch_model.compute_angle(altitude=current_state.y, horizontal_position=current_state.x)
    angle_w_speed = np.pi / 2 + current_state.r - angle_winch_w_vertical

    vectorial_drag = drag*np.array([-np.cos(current_state.r), np.sin(current_state.r)])
    vectorial_lift = lift*np.array([-np.sin(current_state.r), np.cos(current_state.r)])
    vectorial_weight = glider.m * glider.g * np.array([0, -1])
    vectorial_winch = winch_tension*np.array([np.sin(angle_winch_w_vertical), -np.cos(angle_winch_w_vertical)])
    sum_of_forces = vectorial_drag+vectorial_lift+vectorial_weight+vectorial_winch

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
    plt.show()

    pass



def compute_gradients(current_state: States, glider: GliderConstants, winch_model: WinchConstants) -> States:
    if current_state.t % 1 == 0.0:
        generate_forces_in_cartesian(current_state, glider, winch_model)
    drag, lift = glider.compute_drag_and_lift(velocity=current_state.v, attack_angle=current_state.e)
    winch_tension = winch_model.compute_tension(time=current_state.t, distance_from_starting_point=current_state.x)
    angle_winch_w_vertical = winch_model.compute_angle(altitude=current_state.y, horizontal_position=current_state.x)
    angle_w_speed = np.pi/2 + current_state.r - angle_winch_w_vertical

    gradient_v = (-drag * np.cos(current_state.e) - lift * np.sin(current_state.e) -
                  glider.m * glider.g * np.sin(current_state.r) + winch_tension*np.cos(angle_w_speed)) / glider.m
    gradient_r = (-drag * np.sin(current_state.e) + lift * np.cos(current_state.e) - winch_tension*np.sin(angle_w_speed) -
                  glider.m * glider.g * np.cos(current_state.r)) / (glider.m*current_state.v)
    if current_state.y <= 1 and gradient_r < 0:
        gradient_r = 0
    gradient_x = current_state.v * np.cos(current_state.r)
    gradient_y = current_state.v * np.sin(current_state.r)
    return States(v=gradient_v, r=gradient_r, x=gradient_x, y=gradient_y)


def state_transition(current_state: States, gradient_state: States, simulation_parameters: SimulationParameters) \
        -> Tuple[States, bool]:
    v = current_state.v + simulation_parameters.delta_t * gradient_state.v
    r = current_state.r + simulation_parameters.delta_t * gradient_state.r
    x = current_state.x + simulation_parameters.delta_t * gradient_state.x
    y = current_state.y + simulation_parameters.delta_t * gradient_state.y
    t = current_state.t + simulation_parameters.delta_t

    break_constraints = False
    # Check for negative velocity and altitude
    if v < 0:
        v = 0
        break_constraints = True
    if y < 0:
        y = 0
        break_constraints = True

    return States(v=v, r=r, x=x, y=y, t=t), break_constraints


def simulate(glider: GliderConstants, winch_model: WinchConstants, simulation_parameters: SimulationParameters,
             container: SimulationContainer):
    try:
        for t in range(simulation_parameters.n_iterations):
            current_state = container.get_state(time=t)
            grad_state = compute_gradients(current_state, glider, winch_model)
            next_state, meet_constraints = state_transition(current_state, grad_state, simulation_parameters)
            container.add_state(time=t + 1, state=next_state)
            if meet_constraints:
                break
            if next_state.x >= simulation_parameters.max_distance:
                break
    except Exception as e:
        logger.error(f"Simulation error: {e}")
