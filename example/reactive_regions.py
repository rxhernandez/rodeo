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
X_VALS = np.linspace(-1.0, +1.0, 256)
P_VALS = np.linspace(-1.0, +1.0, 256)


# Simple double-well model with V(x) = x⁴/4 - x²/2.
# Inheriting from rodeo.System automatically provides .eq_of_motion(qp, t).
class DoubleWell(rodeo.System):
    def force(self, qp, _t):
        return rodeo.position[qp] - rodeo.position[qp]**3


def main():
    propagator = rodeo.Propagator(DoubleWell(), rodeo.RungeKutta4, TIME_STEP)
    rr = raster_rr(X_VALS, P_VALS, TIME, propagator)
    # Do something useful with the result...


def raster_rr(x_vals, p_vals, t, propagator):
    rr = rodeo.ReactiveRegion(propagator,
        reactant=rodeo.QLessEq(-1.0),
        product=rodeo.QGreaterEq(+1.0),
    )
    worker = functools.partial(rr, t=t)
    args = rodeo.grid(x_vals, p_vals).T

    # rodeo.ReactiveRegion plays nicely with parallelization.
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
