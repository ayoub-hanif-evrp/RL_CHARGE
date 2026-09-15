"""Travel time from the instance velocity ``v``, independent of energy.

EVRPTW-GR Model 2 uses ``travel_time = distance / v``. In every published
file ``v = 1``, so time equals distance in Schneider units. The 60 km/h
physical speed is **not** used here; it exists only inside the kW formula.
"""

from domain.quantities import Distance, TravelTime
from physics.errors import InvalidPhysicsParameterError


def travel_time(distance: Distance, average_velocity: float) -> TravelTime:
    if not isinstance(distance, Distance):
        raise TypeError("distance must be a Distance")
    if average_velocity <= 0:
        raise InvalidPhysicsParameterError(
            f"average velocity must be positive, got {average_velocity}"
        )
    return TravelTime(distance.value / average_velocity)
