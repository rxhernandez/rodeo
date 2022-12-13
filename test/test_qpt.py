# Copyright 2022 Johannes Reiff
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import rodeo


STATES_TIME = np.array([
    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    [0.1, 1.1, 2.1, 3.1, 4.1, 5.1],
    [0.2, 1.2, 2.2, 3.2, 4.2, 5.2],
    [0.3, 1.3, 2.3, 3.3, 4.3, 5.3],
    [0.4, 1.4, 2.4, 3.4, 4.4, 5.4],
])
STATES = np.array([
    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    [0.1, 1.1, 2.1, 3.1, 4.1, 5.1],
    [0.2, 1.2, 2.2, 3.2, 4.2, 5.2],
    [0.3, 1.3, 2.3, 3.3, 4.3, 5.3],
])
POSITIONS = np.array([
    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    [0.1, 1.1, 2.1, 3.1, 4.1, 5.1],
])
MOMENTA = np.array([
    [0.2, 1.2, 2.2, 3.2, 4.2, 5.2],
    [0.3, 1.3, 2.3, 3.3, 4.3, 5.3],
])
TIME = np.array(
    [0.4, 1.4, 2.4, 3.4, 4.4, 5.4],
)


def test_dofs():
    assert rodeo.dofs(STATES_TIME) == 2
    assert rodeo.dofs(STATES) == 2

    assert rodeo.dofs(STATES_TIME.T[0]) == 2
    assert rodeo.dofs(STATES.T[0]) == 2


def test_time():
    assert np.all(rodeo.time[STATES_TIME] == TIME)
    assert rodeo.time[STATES] is None

    assert rodeo.time[STATES_TIME.T[0]] == TIME[0]
    assert rodeo.time[STATES.T[0]] is None


def test_position():
    assert np.all(rodeo.position[STATES_TIME] == POSITIONS)
    assert np.all(rodeo.position[STATES] == POSITIONS)

    assert np.all(rodeo.position[STATES_TIME.T[0]] == POSITIONS.T[0])
    assert np.all(rodeo.position[STATES.T[0]] == POSITIONS.T[0])

    for idx in range(2):
        assert np.all(rodeo.position[STATES_TIME][idx] == POSITIONS[idx])
        assert np.all(rodeo.position[STATES][idx] == POSITIONS[idx])

        assert np.all(rodeo.position[STATES_TIME.T[0]][idx] == POSITIONS[idx, 0])
        assert np.all(rodeo.position[STATES.T[0]][idx] == POSITIONS[idx, 0])


def test_momentum():
    assert np.all(rodeo.momentum[STATES_TIME] == MOMENTA)
    assert np.all(rodeo.momentum[STATES] == MOMENTA)

    assert np.all(rodeo.momentum[STATES_TIME.T[0]] == MOMENTA.T[0])
    assert np.all(rodeo.momentum[STATES.T[0]] == MOMENTA.T[0])

    for idx in range(2):
        assert np.all(rodeo.momentum[STATES_TIME][idx] == MOMENTA[idx])
        assert np.all(rodeo.momentum[STATES][idx] == MOMENTA[idx])

        assert np.all(rodeo.momentum[STATES_TIME.T[0]][idx] == MOMENTA[idx, 0])
        assert np.all(rodeo.momentum[STATES.T[0]][idx] == MOMENTA[idx, 0])


def test_states():
    zero_p = np.zeros_like(MOMENTA)
    zero_t = np.zeros_like(TIME).reshape((1, -1))
    states_zero_p = np.concatenate((POSITIONS, zero_p))
    states_zero_pt = np.concatenate((POSITIONS, zero_p, zero_t))
    states_zero_t = np.concatenate((STATES, zero_t))

    assert np.all(rodeo.states(POSITIONS) == states_zero_p)
    assert np.all(rodeo.states(POSITIONS, t=0.0) == states_zero_pt)
    assert np.all(rodeo.states(POSITIONS, MOMENTA) == STATES)
    assert np.all(rodeo.states(POSITIONS, MOMENTA, t=0.0) == states_zero_t)
    assert np.all(rodeo.states(POSITIONS, MOMENTA, TIME) == STATES_TIME)

    assert np.all(rodeo.states(POSITIONS.T[0]) == states_zero_p.T[0])
    assert np.all(rodeo.states(POSITIONS.T[0], t=0.0) == states_zero_pt.T[0])
    assert np.all(rodeo.states(POSITIONS.T[0], MOMENTA.T[0]) == STATES.T[0])
    assert np.all(rodeo.states(POSITIONS.T[0], MOMENTA.T[0], t=0.0) == states_zero_t.T[0])
    assert np.all(rodeo.states(POSITIONS.T[0], MOMENTA.T[0], TIME[0]) == STATES_TIME.T[0])


def test_grid():
    assert np.all(rodeo.grid([1.0, 2.0, 3.0], [1.1, 2.1, 3.1]) == [
        [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0],
        [1.1, 2.1, 3.1, 1.1, 2.1, 3.1, 1.1, 2.1, 3.1],
    ])
