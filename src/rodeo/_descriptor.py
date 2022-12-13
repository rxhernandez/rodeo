# Copyright 2022 Johannes Reiff
# SPDX-License-Identifier: Apache-2.0

import collections.abc as cabc
import dataclasses
import enum
import typing

import numpy as np
import scipy.integrate

from . import _propagate
from . import _qpt


__all__ = [
    'RegionPredicate',
    'TrajectoryMeasure',
    'ClassificationTimeout',
    'QLess',
    'QLessEq',
    'QGreater',
    'QGreaterEq',
    'ReactiveRegion',
    'Direction',
    'TimeDescriptor',
    'arc_length',
    'LagrangianDescriptor',
]


RegionPredicate: typing.TypeAlias = cabc.Callable[[np.ndarray, float], bool]
"""A predicate determining whether a state is unambiguously reactant/product."""

TrajectoryMeasure: typing.TypeAlias = cabc.Callable[[np.ndarray], float]
"""A measure assigned to a trajectory in phase space."""


class ClassificationTimeout(RuntimeError):
    """Raised when the reactive region cannot be classified."""

    def __init__(self, qp: np.ndarray, t: float, max_time: float):
        super().__init__(f'The reactive region for {tuple(qp)} at time {t} '
            + f'could not be classified within {max_time} time units.')
        self.qp = qp
        self.t = t
        self.max_time = max_time


@dataclasses.dataclass(frozen=True)
class QLess:
    """A predicate checking if position *q* is less than *x*."""

    x: float
    """The position to compare against."""
    q: int = 0
    """The index of the position coordinate to check."""

    def __call__(self, qp, t):
        return _qpt.position[qp][self.q] < self.x


@dataclasses.dataclass(frozen=True)
class QLessEq:
    """A predicate checking if position *q* is less than or equal to *x*."""

    x: float
    """The position to compare against."""
    q: int = 0
    """The index of the position coordinate to check."""

    def __call__(self, qp, t):
        return _qpt.position[qp][self.q] <= self.x


@dataclasses.dataclass(frozen=True)
class QGreater:
    """A predicate checking if position *q* is greater than *x*."""

    x: float
    """The position to compare against."""
    q: int = 0
    """The index of the position coordinate to check."""

    def __call__(self, qp, t):
        return _qpt.position[qp][self.q] > self.x


@dataclasses.dataclass(frozen=True)
class QGreaterEq:
    """A predicate checking if position *q* is greater than or equal to *x*."""

    x: float
    """The position to compare against."""
    q: int = 0
    """The index of the position coordinate to check."""

    def __call__(self, qp, t):
        return _qpt.position[qp][self.q] >= self.x


@dataclasses.dataclass(frozen=True)
class ReactiveRegion:
    """Classify reactive (and nonreactive) regions in phase space."""

    __OUTPUT_ENCODING = {
        (0, 0): 0,
        (0, 1): 1,
        (1, 1): 2,
        (1, 0): 3,
    }

    propagator: _propagate.Propagator
    """How to propagate a state in the system under investigation."""
    reactant: RegionPredicate
    """A predicate determining whether or not a state is a reactant."""
    product: RegionPredicate
    """A predicate determining whether or not a state is a product."""
    max_time: float = np.inf
    """Where to cap the time descriptor to guarantee a finite computing time."""

    def __post_init__(self):
        assert self.max_time > 0.0

    def __call__(self, qp: np.ndarray, t: float) -> int:
        """Determine the reactive region of a state *qp* at time *t*."""

        state_from = self.__classify(qp, t, -abs(self.propagator.dt))
        state_to = self.__classify(qp, t, +abs(self.propagator.dt))
        return self.__OUTPUT_ENCODING[state_from, state_to] # pyright: ignore

    def __classify(self, qp, t, dt):
        initial_time = t
        step = self.propagator.stepper(qp, t, dt)

        while True:
            if self.reactant(qp, t):
                return 0
            if self.product(qp, t):
                return 1
            if abs(t - initial_time) >= self.max_time:
                raise ClassificationTimeout(qp, initial_time, self.max_time)

            qp, t = step()


class Direction(enum.Enum):
    """The time direction a trajectory should be propagated in."""

    BACKWARD = -1
    FORWARD = +1


@dataclasses.dataclass(frozen=True)
class TimeDescriptor:
    """A descriptor based on the time required to leave the saddle region."""

    propagator: _propagate.Propagator
    """How to propagate a state in the system under investigation."""
    predicate: RegionPredicate | cabc.Collection[RegionPredicate]
    """Propagation continues until any of the predicates is :data:`True`."""
    time_direction: Direction
    """Whether to propagate forward or backward in time."""
    max_time: float = np.inf
    """Where to cap the time descriptor to guarantee a finite computing time."""

    def __post_init__(self):
        assert self.max_time > 0.0

        predicate = self.predicate
        if not isinstance(predicate, cabc.Callable):
            object.__setattr__(self, 'predicate', lambda qp, t:
                any(p(qp, t) for p in predicate))

    def __call__(self, qp: np.ndarray, t: float) -> float:
        """Determine the time descriptor of a state *qp* at time *t*."""

        initial_time = t
        dt = self.time_direction.value * abs(self.propagator.dt)
        step = self.propagator.stepper(qp, t, dt)

        while True:
            delta = abs(t - initial_time)
            if self.predicate(qp, t) or delta >= self.max_time:
                return delta

            qp, t = step()


def arc_length(trajectory: np.ndarray) -> float:
    """Return a trajectory's arc length."""

    return abs(scipy.integrate.simpson(
        y=np.linalg.norm(_qpt.momentum[trajectory], axis=0),
        x=_qpt.time[trajectory],
    ))


@dataclasses.dataclass(frozen=True)
class LagrangianDescriptor:
    """A descriptor based on trajectories propagated for a fixed amount of time."""

    propagator: _propagate.Propagator
    """How to propagate a state in the system under investigation."""
    measure: TrajectoryMeasure
    """The measure applied to the final trajectory, e.g., :func:`.arc_length`."""
    time_direction: Direction
    """Whether to propagate forward or backward in time."""
    max_time: float
    """The duration to propagate trajectories for."""

    def __post_init__(self):
        assert self.max_time > 0.0

    def __call__(self, qp: np.ndarray, t: float) -> float:
        """Determine the Lagrangian descriptor of a state *qp* at time *t*."""

        trajectory = _propagate.trajectory_while(
            self.propagator,
            (lambda _, curr_t: abs(curr_t - t) < self.max_time),
            qp, t,
            self.time_direction.value * abs(self.propagator.dt),
        )
        return self.measure(trajectory)
