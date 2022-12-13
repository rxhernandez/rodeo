#!/usr/bin/env python3
# Copyright 2022 Johannes Reiff
# SPDX-License-Identifier: Apache-2.0

import functools
import multiprocessing
import sys

import numpy as np
import tqdm

import rodeo


TIME = 0.0
TIME_STEP = 1e-3
TAU = 8.0
X_VALS = np.linspace(-1.6, +1.6, 256)
P_VALS = np.linspace(-1.0, +1.0, 256)


# Simple double-well model with V(x) = x⁴/4 - x²/2.
# Inheriting from rodeo.System automatically provides `.eq_of_motion(qp, t)`.
class DoubleWell(rodeo.System):
    def force(self, qp, _t):
        return rodeo.position[qp] - rodeo.position[qp]**3


def main():
    propagator = rodeo.Propagator(DoubleWell(), rodeo.RungeKutta4, TIME_STEP)
    fwd_ld = raster_ld(X_VALS, P_VALS, TIME, propagator, rodeo.Direction.FORWARD, TAU)
    bwd_ld = raster_ld(X_VALS, P_VALS, TIME, propagator, rodeo.Direction.BACKWARD, TAU)
    ld = fwd_ld + bwd_ld
    # Do something useful with the result...


def raster_ld(x_vals, p_vals, t, propagator, time_direction, max_time):
    ld = rodeo.LagrangianDescriptor(propagator,
        measure=rodeo.arc_length,
        time_direction=time_direction,
        max_time=max_time,
    )

    worker = functools.partial(ld, t=t)
    args = rodeo.grid(x_vals, p_vals).T

    # rodeo.LagrangianDescriptor plays nicely with parallelization.
    # tqdm.tqdm provides a progress bar.
    # np.array(...).reshape() turns the list of results into a 2D array
    # that can be plottet with, e.g., Matplotlib's .imshow().
    with multiprocessing.Pool() as pool:
        return np.array(list(
            tqdm.tqdm(
                pool.imap(worker, args, chunksize=16),
                total=len(args), miniters=1,
            )
        )).reshape(len(x_vals), len(p_vals))


if __name__ == '__main__':
    sys.exit(int(main() or 0))
