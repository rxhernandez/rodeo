# Copyright 2022 Johannes Reiff
# SPDX-License-Identifier: Apache-2.0

import collections.abc as cabc
import contextlib
import dataclasses
import typing

import numpy as np
import numpy.typing as npt

from . import _qpt


__all__ = [
    'RegionClassifier',
    'QuadInitializer',
    'CannotFindQuad',
    'NoConvergence',
    'ContractionFailed',
    'BinaryContraction',
    'CrossQuadInit',
    'EllipseQuadInit',
]


RegionClassifier: typing.TypeAlias = cabc.Callable[[np.ndarray, float], int]
"""A callable classifying reactive (and nonreactive) regions in phase space."""

QuadInitializer: typing.TypeAlias = cabc.Callable[
    [RegionClassifier, np.ndarray, float], np.ndarray]
"""A quadrangle initialization routine for the BCM."""


class CannotFindQuad(RuntimeError):
    """Raised by quadrangle initializers if their heuristic fails."""

    def __init__(self):
        super().__init__('Failed to find the initial quadrangle for the BCM.')


class NoConvergence(RuntimeError):
    """Signals that the BCM did not converge to the given tolerance."""

    def __init__(self, max_iters: int):
        super().__init__(f'Failed to converge after {max_iters} iterations.')
        self.max_iters = max_iters


class ContractionFailed(RuntimeError):
    """Signals that the BCM aborted due to unexpected phase space structure."""

    def __init__(self):
        super().__init__('Failed to contract due to unexpected phase space structure.')


@dataclasses.dataclass(frozen=True)
class BinaryContraction:
    """The BCM [Barda2018]_ projects a state to the NHIM along the reaction coordinate."""

    classify: RegionClassifier
    """A region classifier, typically an instance of :class:`ReactiveRegion`."""
    initializers: cabc.Collection[QuadInitializer]
    """A collection of callables used to initialize the quadrangle."""
    tol: float = 1e-8
    """The algorithm's tolerance, i.e., a measure for the final quadrangle size."""
    max_iters: int = 64
    """The maximum number of iterations before giving up."""

    def __post_init__(self):
        assert self.tol > 0.0
        assert self.max_iters > 0

    def __call__(self, qp: npt.ArrayLike, t: float) -> tuple[np.ndarray, float]:
        """
        Project the state *qp* at time *t* to the NHIM.

        :returns: The projected state and an error estimate.
        """

        quad = self.__init_quad(np.asarray(qp), t)
        iters = 0

        while (error := self.__error(quad)) > self.tol:
            if (iters := iters + 1) > self.max_iters:
                raise NoConvergence(self.max_iters)
            self.__contract(quad, t)

        return self.__center(quad), error

    def __init_quad(self, qp, t):
        for initializer in self.initializers:
            with contextlib.suppress(CannotFindQuad):
                return initializer(self.classify, qp, t)
        raise CannotFindQuad

    def __contract(self, quad, t):
        for i in range(4):
            midpoint = 0.5 * (quad[i - 1] + quad[i])
            region = self.classify(midpoint, t)
            if region in ((i - 1) % 4, i):
                quad[region] = midpoint
            else:
                raise ContractionFailed

    @staticmethod
    def __center(quad):
        return np.mean(quad, axis=0)

    @staticmethod
    def __error(quad):
        """Error estimate is currently the length of the longest edge."""
        return np.linalg.norm(quad - np.roll(quad, 1, axis=0), axis=1).max()


@dataclasses.dataclass(frozen=True)
class CrossQuadInit:
    """Initialize the corners of the BCM quadrangle along a cross shape."""

    delta_q: float
    """The step size in position space."""
    delta_p: float
    """The step size in momentum space."""
    max_iters_q: int = 16
    """The maximum number of steps in position space to try."""
    max_iters_p: int = 16
    """The maximum number of steps in momentum space to try."""

    def __post_init__(self):
        assert self.delta_q > 0.0
        assert self.delta_p > 0.0
        assert self.max_iters_q > 0
        assert self.max_iters_p > 0

    def __call__(self,
        classify: RegionClassifier,
        qp: np.ndarray,
        t: float,
    ) -> np.ndarray:

        """Try to find good initial conditions for the BCM quadrangle."""

        return np.array([
            self.__find(classify, qp, t, 0, -self.delta_q, self.max_iters_q, _qpt.position),
            self.__find(classify, qp, t, 1, +self.delta_p, self.max_iters_p, _qpt.momentum),
            self.__find(classify, qp, t, 2, +self.delta_q, self.max_iters_q, _qpt.position),
            self.__find(classify, qp, t, 3, -self.delta_p, self.max_iters_p, _qpt.momentum),
        ])

    @staticmethod
    def __find(classify, qp, t, region, delta, max_iters, coord):
        qp = np.copy(qp)
        for _ in range(max_iters):
            coord[qp][0] += delta
            if classify(qp, t) == region:
                return qp

        raise CannotFindQuad


@dataclasses.dataclass(frozen=True)
class EllipseQuadInit:
    """Initialize the corners of the BCM quadrangle along an ellipse [Reiff2021]_."""

    radius_q: float
    """The semidiameter of the ellipse in position space."""
    radius_p: float
    """The semidiameter of the ellipse in momentum space."""
    max_iters: int = 256
    """The maximum number of steps to try."""

    def __post_init__(self):
        assert self.radius_q > 0.0
        assert self.radius_p > 0.0
        assert self.max_iters > 0

    def __call__(self,
        classify: RegionClassifier,
        center_qp: np.ndarray,
        t: float,
    ) -> np.ndarray:

        """Try to find good initial conditions for the BCM quadrangle."""

        def rotated(angle):
            new = np.copy(center_qp)
            _qpt.position[new][0] -= self.radius_q * np.cos(angle)
            _qpt.momentum[new][0] += self.radius_p * np.sin(angle)
            return new, classify(new, t)

        quad = np.empty((4, len(center_qp)))
        last_angle = 0.0
        qp, last_region = rotated(last_angle)
        quad[last_region] = qp

        for _ in range(3):
            angle_step = 0.25 * np.pi

            for _ in range(self.max_iters):
                qp, region = rotated(last_angle + angle_step)
                if region == last_region:
                    last_angle += angle_step
                elif (region - last_region) % 4 == 1:
                    quad[region] = qp
                    last_region = region
                    break
                else:
                    angle_step *= 0.5
            else:
                raise CannotFindQuad

        return quad
