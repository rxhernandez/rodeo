# Copyright 2022 Johannes Reiff
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import rodeo


GRAD_TOLERANCE = 1e-8
GRAD_EPSILON = 1e-6
POINTS_PER_DOF = 1000
SYSTEMS = {
    'Model2dAtan': (
        rodeo.Model2dAtan(
            barr_height=2.0,
            barr_width=1.0 / np.sqrt(2.0),
            osc_amplitude=0.4,
            osc_frequency=np.pi,
            orth_frequency=2.0,
            atan_amplitude=1.0,
            atan_slope=2.0,
        ),
        [(-1.5, +1.5), (-1.5, +1.5)],
    ),
}


@pytest.mark.parametrize(['sys', 'extent'], SYSTEMS.values(), ids=SYSTEMS.keys())
def test_force(sys, extent):
    states = states_on_grid(extent)
    analytical = sys.force(states, t=0.0)
    numerical = grad_force(sys, states, t=0.0)

    # pytest.approx is very slow for large arrays
    assert np.abs(analytical - numerical).max() <= GRAD_TOLERANCE


def test_model2datan():
    sys = SYSTEMS['Model2dAtan'][0]
    pot = lambda qp, t: sys.potential(np.array(qp), t)

    assert pot([0.0, 0.0, 0.0, 0.0], 0.0) == sys.barr_height
    assert pot([0.0, 0.0, 0.0, 0.0], 0.5) < sys.barr_height

    assert pot([+1.0, +1.0, 0.0, 0.0], 0.0) < sys.barr_height
    assert pot([-1.0, -1.0, 0.0, 0.0], 0.0) < sys.barr_height
    assert pot([-1.0, +1.0, 0.0, 0.0], 0.0) > sys.barr_height
    assert pot([+1.0, -1.0, 0.0, 0.0], 0.0) > sys.barr_height

    val = 0.5 * sys.orth_frequency**2 * sys.atan_amplitude**2
    assert pot([-np.inf, 0.0, 0.0, 0.0], 0.0) == val
    assert pot([+np.inf, 0.0, 0.0, 0.0], 0.0) == val
    assert pot([-np.inf, -2.0 * sys.atan_amplitude, 0.0, 0.0], 0.0) == val
    assert pot([+np.inf, +2.0 * sys.atan_amplitude, 0.0, 0.0], 0.0) == val
    assert pot([-np.inf, -sys.atan_amplitude, 0.0, 0.0], 0.0) == 0.0
    assert pot([+np.inf, +sys.atan_amplitude, 0.0, 0.0], 0.0) == 0.0


def states_on_grid(extent):
    ranges = (np.linspace(*lim, POINTS_PER_DOF) for lim in extent)
    return rodeo.states(q=rodeo.grid(*ranges))


def grad_force(sys, states, t, eps=GRAD_EPSILON):
    n = rodeo.dofs(states)
    pot = lambda i, delta: sys.potential(states + np.eye(1, 2 * n, i).T * delta, t)

    return np.stack(
        [(pot(i, -eps) - pot(i, +eps)) / (2 * eps) for i in range(n)],
    )
