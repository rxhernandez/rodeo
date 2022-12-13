# Copyright 2022 Johannes Reiff
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=redefined-outer-name

import numpy as np
import pytest

import rodeo


ORIGIN_X = 0.51 # Reduce risk of accidentally hitting the NHIM exactly.
QUAD_INIT = {
    'CrossQuadInit': rodeo.CrossQuadInit(delta_q=0.2, delta_p=0.2),
    'EllipseQuadInit': rodeo.EllipseQuadInit(radius_q=1.0, radius_p=1.0),
}


class HarmonicSaddle(rodeo.System):
    def force(self, qp: np.ndarray, t: float) -> np.ndarray:
        n = (rodeo.dofs(qp) - 1)
        return ([+1] + n * [-1]) * (rodeo.position[qp] - ORIGIN_X)


@pytest.fixture
def classifier():
    return rodeo.ReactiveRegion(
        rodeo.Propagator(HarmonicSaddle(), rodeo.RungeKutta4, dt=1e-3),
        reactant=rodeo.QLessEq(-1.0),
        product=rodeo.QGreaterEq(+3.0),
        max_time=100.0,
    )


@pytest.mark.slow
@pytest.mark.parametrize('quad_init', QUAD_INIT.values(), ids=QUAD_INIT.keys())
@pytest.mark.parametrize(['qp', 'expected'], [
    ([0.0, 0.0], [ORIGIN_X, 0.0]),
    ([0.0, 0.5], [ORIGIN_X, 0.0]),
    ([0.0, 1.0, 0.0, 2.0], [ORIGIN_X, 1.0, 0.0, 2.0]),
    ([0.5, 2.0, 0.5, 1.0], [ORIGIN_X, 2.0, 0.0, 1.0]),
])
def test_binary_contraction(classifier, quad_init, qp, expected):
    bcm = rodeo.BinaryContraction(classifier, [quad_init])
    _assert_bcm(bcm, qp, expected)


@pytest.mark.slow
def test_binary_contraction_multi_init(classifier):
    bcm = rodeo.BinaryContraction(classifier, [
        rodeo.CrossQuadInit(delta_q=0.1, delta_p=0.1, max_iters_q=2),
        rodeo.CrossQuadInit(delta_q=0.2, delta_p=0.2, max_iters_q=4),
    ])
    _assert_bcm(bcm, [0.0, 0.0], [ORIGIN_X, 0.0])


@pytest.mark.parametrize('quad_init', QUAD_INIT.values(), ids=QUAD_INIT.keys())
def test_binary_contraction_init_failure(classifier, quad_init):
    bcm = rodeo.BinaryContraction(classifier, [quad_init])
    with pytest.raises(rodeo.CannotFindQuad):
        bcm([-4.0, 0.0], 0.0)


def test_binary_contraction_max_iters(classifier):
    bcm = rodeo.BinaryContraction(
        classifier, QUAD_INIT.values(), tol=1e-12, max_iters=4)
    with pytest.raises(rodeo.NoConvergence):
        bcm([0.0, 0.0], 0.0)


def _assert_bcm(bcm, qp, expected):
    actual, error = bcm(qp, 0.0)
    assert error <= bcm.tol
    assert actual == pytest.approx(expected, rel=0.0, abs=2 * bcm.tol)
