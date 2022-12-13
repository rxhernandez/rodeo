# Copyright 2022 Johannes Reiff
# SPDX-License-Identifier: Apache-2.0

import collections.abc as cabc
import dataclasses
import typing

import numpy as np
import numpy.typing as npt

from . import _qpt
from . import _system


__all__ = [
    'Stepper',
    'TrajectoryPredicate',
    'StepperFactory',
    'Propagator',
    'trajectory_while',
    'VelocityVerlet',
    'RungeKutta4',
]


Stepper: typing.TypeAlias = cabc.Callable[[], tuple[np.ndarray, float]]
"""A protocol representing the integration algorithm."""

TrajectoryPredicate: typing.TypeAlias = cabc.Callable[[np.ndarray, float], bool]
"""A predicate determining whether a trajectory should continue to propagate."""


class StepperFactory(typing.Protocol):
    """A protocol representing stepper factories"""

    def __call__(self,
        system: _system.System,
        qp: np.ndarray,
        t: float,
        dt: float,
    ) -> Stepper:
        ...


@dataclasses.dataclass(frozen=True)
class Propagator:
    """Hold the information required to propagate a state in a given system."""

    system: _system.System
    """The physical system to propagate in."""
    make_stepper: StepperFactory
    """A factory function (e.g., a class) for a concrete stepper."""
    dt: float
    """The default integration time step."""
    args: dict = dataclasses.field(default_factory=dict)
    """Additional keyword arguments to pass to the stepper constructor."""

    def stepper(self,
        qp: np.ndarray,
        t: float,
        dt: float | None = None,
    ) -> Stepper:

        """Construct a stepper object propagating state *qp* starting at time *t*."""

        dt = self.dt if dt is None else dt
        return self.make_stepper(self.system, qp, t, dt, **self.args)


def trajectory_while(
    propagator: Propagator,
    predicate: TrajectoryPredicate,
    qp: npt.ArrayLike,
    t: float,
    dt: float | None = None,
) -> np.ndarray:

    """Propagate state *qp* while *predicate* holds true and return the trajectory."""

    qp = np.asarray(qp)
    step = propagator.stepper(qp, t, dt)
    trajectory = [(*qp, t)]

    while predicate(qp, t):
        qp, t = step()
        trajectory.append((*qp, t))

    return np.array(trajectory).T


class VelocityVerlet:
    """Implementation of the Velocity Verlet symplectic integration scheme."""

    def __init__(self,
        system: _system.System,
        qp: np.ndarray,
        t: float,
        dt: float,
    ):
        self.__system = system
        self.__qp = np.copy(qp)
        self.__t = t
        self.__dt = dt
        self.__force = system.force(qp, t)

    def __call__(self) -> tuple[np.ndarray, float]:
        """Propagate the state *qp* by one time step *dt*."""

        _qpt.momentum[self.__qp] += 0.5 * self.__dt * self.__force
        _qpt.position[self.__qp] += _qpt.momentum[self.__qp] * self.__dt
        self.__t += self.__dt
        self.__force = self.__system.force(self.__qp, self.__t)
        _qpt.momentum[self.__qp] += 0.5 * self.__dt * self.__force

        return np.copy(self.__qp), self.__t


class RungeKutta4:
    """Implementation of a fourth-order Runge Kutta integration scheme."""

    def __init__(self,
        system: _system.System,
        qp: np.ndarray,
        t: float,
        dt: float,
    ):
        self.__system = system
        self.__qp = np.copy(qp)
        self.__t = t
        self.__dt = dt

    def __call__(self) -> tuple[np.ndarray, float]:
        """Propagate the state *qp* by one time step *dt*."""

        qp, t, dt = self.__qp, self.__t, self.__dt
        k1 = dt * self.__system.eq_of_motion(qp, t)
        k2 = dt * self.__system.eq_of_motion(qp + 0.5 * k1, t + 0.5 * dt)
        k3 = dt * self.__system.eq_of_motion(qp + 0.5 * k2, t + 0.5 * dt)
        k4 = dt * self.__system.eq_of_motion(qp + k3, t + dt)

        self.__qp += (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        self.__t += dt

        return np.copy(self.__qp), self.__t
