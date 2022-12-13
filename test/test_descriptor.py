# Copyright 2022 Johannes Reiff
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=redefined-outer-name

import numpy as np
import pytest

import rodeo


class HarmonicSaddle(rodeo.System):
    def force(self, qp: np.ndarray, t: float) -> np.ndarray:
        return ([+1] + (rodeo.dofs(qp) - 1) * [-1]) * rodeo.position[qp]


@pytest.fixture
def propagator():
    return rodeo.Propagator(HarmonicSaddle(), rodeo.RungeKutta4, dt=1e-3)


def test_reactive_region(propagator):
    classify = rodeo.ReactiveRegion(propagator,
        reactant=rodeo.QLessEq(-1.0),
        product=rodeo.QGreaterEq(+1.0),
    )

    assert classify(np.array([-0.5,  0.0           ]), 0.0) == 0
    assert classify(np.array([-0.5,  0.0,  0.0, 0.0]), 0.0) == 0
    assert classify(np.array([ 0.0, +0.5           ]), 0.0) == 1
    assert classify(np.array([ 0.0,  0.0, +0.5, 0.0]), 0.0) == 1
    assert classify(np.array([+0.5,  0.0           ]), 0.0) == 2
    assert classify(np.array([+0.5,  0.0,  0.0, 0.0]), 0.0) == 2
    assert classify(np.array([ 0.0, -0.5           ]), 0.0) == 3
    assert classify(np.array([ 0.0,  0.0, -0.5, 0.0]), 0.0) == 3


def test_reactive_region_timeout(propagator):
    classify = rodeo.ReactiveRegion(propagator,
        reactant=rodeo.QLessEq(-1.0),
        product=rodeo.QGreaterEq(+1.0),
        max_time=1.0,
    )

    with pytest.raises(rodeo.ClassificationTimeout):
        classify(np.array([-0.5, 0.0          ]), 0.0)
    with pytest.raises(rodeo.ClassificationTimeout):
        classify(np.array([-0.5, 0.0, 0.0, 0.0]), 0.0)

    assert classify(np.array([-0.8, 0.0          ]), 0.0) == 0
    assert classify(np.array([-0.8, 0.0, 0.0, 0.0]), 0.0) == 0


def test_time_descriptor(propagator):
    _test_descriptor(
        numerical=rodeo.TimeDescriptor(propagator,
            predicate=[rodeo.QLessEq(-1.0), rodeo.QGreaterEq(+1.0)],
            time_direction=rodeo.Direction.FORWARD,
            max_time=2.0,
        ),
        analytical=lambda qp: min(_harm_barr_time_desc(qp, 1.0), 2.0),
        tolerance=2 * propagator.dt,
    )

    _test_descriptor(
        numerical=rodeo.TimeDescriptor(propagator,
            predicate=[rodeo.QLessEq(-1.0), rodeo.QGreaterEq(+1.0)],
            time_direction=rodeo.Direction.BACKWARD,
            max_time=2.0,
        ),
        analytical=lambda qp: min(_harm_barr_time_desc(qp, 1.0), 2.0),
        tolerance=2 * propagator.dt,
    )


def test_time_descriptor_single_predicate(propagator):
    def pred(qp, _):
        x = rodeo.position[qp][0]
        return x <= -1.0 or x >= +1.0

    _test_descriptor(
        numerical=rodeo.TimeDescriptor(propagator,
            predicate=pred,
            time_direction=rodeo.Direction.FORWARD,
            max_time=2.0,
        ),
        analytical=lambda qp: min(_harm_barr_time_desc(qp, 1.0), 2.0),
        tolerance=2 * propagator.dt,
    )


def test_lagrangian_descriptor(propagator):
    _test_descriptor(
        numerical=rodeo.LagrangianDescriptor(propagator,
            measure=rodeo.arc_length,
            time_direction=rodeo.Direction.FORWARD,
            max_time=2.0,
        ),
        analytical=lambda qp: _harm_barr_lagn_desc(qp, 2.0),
        tolerance=2 * propagator.dt,
    )

    _test_descriptor(
        numerical=rodeo.LagrangianDescriptor(propagator,
            measure=rodeo.arc_length,
            time_direction=rodeo.Direction.BACKWARD,
            max_time=2.0,
        ),
        analytical=lambda qp: _harm_barr_lagn_desc(qp, -2.0),
        tolerance=2 * propagator.dt,
    )


def _test_descriptor(numerical, analytical, tolerance):
    for qp in np.linspace((-1.0, 0.0), (+1.0, 0.0), 21):
        assert numerical(qp, 0.0) == pytest.approx(analytical(qp), tolerance)
    for qp in np.linspace((0.0, -1.0), (0.0, +1.0), 21):
        assert numerical(qp, 0.0) == pytest.approx(analytical(qp), tolerance)


def _harm_barr_time_desc(qp_0, region):
    if qp_0[0] <= -region or qp_0[0] >= +region:
        return 0.0

    if np.all(qp_0 == 0.0):
        return np.inf
    elif qp_0[0] == 0.0:
        return np.arcsinh(region / abs(qp_0[1]))
    elif qp_0[1] == 0.0:
        return np.arccosh(region / abs(qp_0[0]))
    else:
        raise NotImplementedError


def _harm_barr_lagn_desc(qp_0, t):
    return abs(qp_0[0] * np.cosh(t) + qp_0[1] * np.sinh(t) - qp_0[0])
