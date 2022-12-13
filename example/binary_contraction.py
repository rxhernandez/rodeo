#!/usr/bin/env python3
# Copyright 2022 Johannes Reiff
# SPDX-License-Identifier: Apache-2.0

import sys

import numpy as np

import rodeo


def main():
    # Configure the system (standard built-in test model):
    system = rodeo.Model2dAtan(
        barr_height=2.0,
        barr_width=1.0 / np.sqrt(2.0),
        osc_amplitude=0.4,
        osc_frequency=np.pi,
        orth_frequency=2.0,
        atan_amplitude=1.0,
        atan_slope=2.0,
    )

    # Configure the BCM:
    classifier = rodeo.ReactiveRegion(
        rodeo.Propagator(system, rodeo.VelocityVerlet, dt=1e-3),
        reactant=rodeo.QLessEq(-1.0),
        product=rodeo.QGreaterEq(+1.0),
        max_time=100.0,
    )
    initializers = [
        rodeo.CrossQuadInit(delta_q=0.2, delta_p=0.2),
        rodeo.EllipseQuadInit(radius_q=1.0, radius_p=1.0),
    ]
    bcm = rodeo.BinaryContraction(classifier, initializers)

    # Run the BCM for y = 0.2, p_y = 0.4 with initial guess x = 0.1, p_x = 0.3:
    projected_qp, estimated_error = bcm(qp=[0.1, 0.2, 0.3, 0.4], t=0.0)
    # Do something useful with the result...



if __name__ == '__main__':
    sys.exit(int(main() or 0))
