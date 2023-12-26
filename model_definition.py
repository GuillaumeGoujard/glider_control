from typing import List

import numpy as np
import attr

def kmh_to_ms(velocity_kmh: float) -> float:
    """Convert velocity from km/h to m/s."""
    return velocity_kmh / 3.6


def round_float(value: float) -> float:
    """Round a float value to two decimal places."""
    return round(value, 2)

# Functions to estimate drag and lift coefficients as functions of alpha
def drag_coefficient(alpha_deg):
    # Example estimation, you can replace with your own function or data
    alpha_rad = np.deg2rad(alpha_deg)
    return 0.02 + 0.05 * alpha_rad

def lift_coefficient(alpha_deg):
    # Example estimation, you can replace with your own function or data
    alpha_rad = np.deg2rad(alpha_deg)
    return 0.3 + 0.4 * alpha_rad


@attr.s(auto_attribs=True, frozen=True)
class GliderConstants:
    """
    Class to hold and manage constants for a glider.
    """
    m: float = 300.0  # Mass in kg
    g: float = 9.81  # Gravitational acceleration in m/s^2
    rho: float = 1.3  # Air density in kg/m^3
    S: float = 15.0  # Wing area in m^2

    def compute_drag_and_lift(self, velocity: float, attack_angle:float) -> tuple[float, float]:
        """
        Compute the drag and lift forces for a given velocity.

        :param velocity: Velocity in m/s
        :return: Tuple containing drag force (Px) and lift force (Pz)
        """
        Px = 0.5 * drag_coefficient(attack_angle) * self.rho * self.S * np.square(velocity)
        Pz = 0.5 * lift_coefficient(attack_angle) * self.rho * self.S * np.square(velocity)
        return Px, Pz


@attr.s(auto_attribs=True, frozen=True)
class WinchConstants:
    """
    Class to hold and manage constants for a glider.
    """
    distance_from_take_off: int = 1000 # meters
    percent_distance_to_drop: int = 80 # meters
    min_tension: float = 4000 # Newton
    max_tension: float = 8000 # Newton
    rate_of_change: float = 500 # Newton/s
    target_speed_cable: float = 30 # m/s
    K_p: float = 10
    K_i: float = 1

    def compute_tension(self, time: float, distance_from_starting_point: float) -> float:
        if distance_from_starting_point >= self.percent_distance_to_drop*self.distance_from_take_off:
            return 0
        else:
            return min(self.max_tension, self.min_tension + self.rate_of_change*time)

    # def compute_tension(self, actual_speed_cable: float, dt: float, total_error: float):
    #     """
    #     Compute the tension using a PID controller.
    #
    #     :param actual_speed_cable: The actual speed of the cable in m/s
    #     :param dt: Time step in seconds
    #     :param total_error: The cumulative speed error
    #     :return: The tension in Newtons
    #     """
    #     # Calculate the speed error
    #     error = self.target_speed_cable - actual_speed_cable
    #
    #     # Proportional term
    #     P = self.K_p * error
    #
    #     # Integral term (approximate integral by summing errors and multiplying by time step)
    #     total_error += error * dt
    #     I = self.K_i * total_error
    #
    #     # PID output
    #     tension = P + I
    #     # Enforce minimum and maximum tension limits
    #     tension = max(self.min_tension, min(tension, self.max_tension))
    #
    #     return tension, total_error

    def compute_angle(self, altitude: float, horizontal_position: float):
        if altitude == 0:
            return np.pi/2
        return np.arctan((self.distance_from_take_off-horizontal_position)/altitude)

@attr.s(auto_attribs=True, frozen=True)
class States:
    """
    Class to manage the state of the glider.
    """
    x: float = attr.ib(converter=round_float)
    y: float = attr.ib(converter=round_float)
    v: float = attr.ib(converter=round_float) # velocity in m/s
    r: float = attr.ib(converter=round_float) # flight path angle in radians
    t: float = attr.ib(default=0.0, converter=round_float) # time in second
    e: float = attr.ib(default=0.0) # Attack Angle in radians
    # theta: float = 0.0 # Angle between winch and y axis



@attr.s(auto_attribs=True)
class SimulationParameters:
    t_final: int = 100
    delta_t: float = 0.1
    max_distance: float = 1000
    simulation_timesteps: np.ndarray = attr.ib(init=False)
    n_iterations: int = attr.ib(init=False)

    def __attrs_post_init__(self):
        self.simulation_timesteps = np.arange(0, self.t_final + self.delta_t, self.delta_t)
        self.n_iterations = self.simulation_timesteps.shape[0]



@attr.s(auto_attribs=True)
class SimulationContainer:
    initial_conditions: States
    _states: dict[float, States] = attr.Factory(dict)

    def __attrs_post_init__(self):
        """Initialize the states dictionary with the initial conditions."""
        self._states[0] = self.initial_conditions

    def add_state(self, time: float, state: States):
        """Add a state at a specific time."""
        self._states[time] = state

    def get_state(self, time: float) -> States:
        """Retrieve the state at a specific time."""
        return self._states.get(time, None)

    @property
    def index(self) -> List[int]:
        return list(self._states.keys())

