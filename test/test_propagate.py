# Copyright 2022 Johannes Reiff
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import rodeo


TIME_STEP = 1e-3
STEPPERS = {
    'VelocityVerlet': (rodeo.VelocityVerlet, 6e-7),
    'RungeKutta4': (rodeo.RungeKutta4, 1e-12),
}


class HarmonicOscillator(rodeo.System):
    def force(self, qp: np.ndarray, t: float) -> np.ndarray:
        return -rodeo.position[qp]


@pytest.mark.parametrize(['stepper_type', 'tolerance'],
    STEPPERS.values(), ids=STEPPERS.keys())
def test_trajectory(stepper_type, tolerance):
    propagator = rodeo.Propagator(HarmonicOscillator(), stepper_type, TIME_STEP)
    period = 2 * np.pi

    numerical = rodeo.trajectory_while(
        propagator,
        lambda qp, t: t < period,
        [2.0, 0.0, 0.0, 1.0],
        0.0,
    )
    t = rodeo.time[numerical]
    analytical = np.vstack((
        [[0.0], [1.0]] * np.sin(t) + [[2.0], [0.0]] * np.cos(t),
        [[0.0], [1.0]] * np.cos(t) - [[2.0], [0.0]] * np.sin(t),
        t,
    )) # pyright: ignore

    assert period <= t[-1] < period + TIME_STEP
    assert numerical.shape == (5, int(np.ceil(period / TIME_STEP)) + 1)
    # pytest.approx is very slow for large arrays
    assert np.abs(analytical - numerical).max() <= tolerance
