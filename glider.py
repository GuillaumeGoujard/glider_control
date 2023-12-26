from enum import Enum

import numpy as np
import matplotlib.pyplot as plt
import attr
import logging

from dynamics import simulate
from model_definition import GliderConstants, kmh_to_ms, SimulationParameters, States, SimulationContainer
from modes import get_initial_conditions_from_mode, Mode
from plotting import plot_simulation_results

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

glider = GliderConstants()
simulation_parameters = SimulationParameters(t_final=100, delta_t=0.1, max_distance=1000)

# Select the mode here, and use it to set initial conditions
selected_mode = Mode.GLIDER_GROUNDED_WINCH
initial_conditions, winch_model = get_initial_conditions_from_mode(selected_mode)


container = SimulationContainer(initial_conditions=initial_conditions)

simulate(glider, winch_model, simulation_parameters, container)
plot_simulation_results(winch_model, container)