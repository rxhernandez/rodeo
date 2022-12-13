# Copyright 2022 Johannes Reiff
# SPDX-License-Identifier: Apache-2.0

import abc
import dataclasses
import typing

import numpy as np

from . import _qpt


__all__ = [
    'System',
    'Model2dAtan',
]


class System(typing.Protocol):
    """A protocol representing a physical system."""

    @abc.abstractmethod
    def force(self, qp: np.ndarray, t: float) -> np.ndarray:
        """The force :math:`\\vec{F}` acting on the system state *qp*."""

        raise NotImplementedError

    def eq_of_motion(self, qp: np.ndarray, t: float) -> np.ndarray:
        r"""
        The full equations of motion.

        The default implementation assumes that
        :math:`(\dot{\vec{q}}, \dot{\vec{p}}) = (\vec{p} / m, \vec{F})`
        with mass :math:`m` equal to 1.
        """

        return np.concatenate((_qpt.momentum[qp], self.force(qp, t)))


@dataclasses.dataclass(frozen=True)
class Model2dAtan(System):
    r"""
    A simple 2d model system with a Gaussian barrier in an arctangent valley.

    .. math::

        V(x, y, t) =
            E_\mathrm{b} \exp\left\{-[x - \hat{x} \sin(\omega_x t)]^2 / (2 a^2)\right\}
            + \frac{\omega_y^2}{2}
                \left[y - \frac{2}{\pi} \hat{y} \arctan(\tilde{a} x)\right]^2
    """

    barr_height: float
    """Barrier height :math:`E_\\mathrm{b}`."""
    barr_width: float
    """Barrier width :math:`a`."""
    osc_amplitude: float
    """Barrier driving amplitude :math:`\\hat{x}`."""
    osc_frequency: float
    """Barrier driving angular velocity :math:`\\omega_x`."""
    orth_frequency: float
    """Eigenfrequency :math:`\\omega_y` of the orthogonal mode."""
    atan_amplitude: float
    """Maximum deviation :math:`\\hat{y}` of the minimum energy path from :math:`y = 0`."""
    atan_slope: float
    """Slope :math:`\\tilde{a}`of the arctangent at :math:`x = 0`."""

    def potential(self, qp: np.ndarray, t: float) -> np.ndarray:
        x, y = _qpt.position[qp]
        x_barr = (x - self.osc_amplitude * np.sin(self.osc_frequency * t)) / self.barr_width
        atan_scale = 2 / np.pi * self.atan_amplitude

        return (
            self.barr_height * np.exp(-0.5 * x_barr**2)
            + 0.5 * self.orth_frequency**2
                * (y - atan_scale * np.arctan(self.atan_slope * x))**2
        )

    def force(self, qp: np.ndarray, t: float) -> np.ndarray:
        x, y = _qpt.position[qp]
        x_barr = (x - self.osc_amplitude * np.sin(self.osc_frequency * t)) / self.barr_width
        atan_scale = 2 / np.pi * self.atan_amplitude
        y_mep = self.orth_frequency**2 * (y - atan_scale * np.arctan(self.atan_slope * x))

        return np.array((
            self.barr_height / self.barr_width * x_barr * np.exp(-0.5 * x_barr**2)
                + atan_scale * y_mep * self.atan_slope
                     / ((self.atan_slope * x)**2 + 1),
            -y_mep,
        ))
