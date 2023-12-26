from enum import Enum
from model_definition import States, kmh_to_ms, WinchConstants
import numpy as np

# Extend the existing code by defining an Enum for modes
class Mode(Enum):
    GLIDER_1000M_100KMH = ('1000m', '100km/h')
    GLIDER_GROUNDED_WINCH = ('0m', '0km/h')

# Function to convert the Mode Enum into actual initial conditions
def get_initial_conditions_from_mode(mode: Mode):
    if mode == Mode.GLIDER_1000M_100KMH:
        return States(x=0, y=1000.0, v=kmh_to_ms(100), r=0.0), WinchConstants(max_tension=0)
    elif mode == Mode.GLIDER_GROUNDED_WINCH:
        return States(x=0, y=0.0, v=kmh_to_ms(0), r=0.0), WinchConstants()
    else:
        raise ValueError("Unknown mode")