#!/usr/bin/env python3

import functools
import multiprocessing
import pickle
import sys

import numpy as np
import tqdm

import rodeo


TIME = 0.0
TIME_STEP = 1e-3
TAU = 8.0
X_VALS = np.linspace(-1.6, +1.6, 1024)
P_VALS = np.linspace(-1.0, +1.0, 1024)


class DoubleWell(rodeo.System):
    def force(self, qp, _t):
        return rodeo.position[qp] - rodeo.position[qp]**3


def main():
    propagator = rodeo.Propagator(DoubleWell(), rodeo.RungeKutta4, TIME_STEP)
    raster = lambda t_dir: raster_ld(X_VALS, P_VALS, TIME, propagator, t_dir, TAU)

    with open('ld.pickle', 'wb') as fp:
        pickle.dump({
            'system': 'V = x⁴/4 - x²/2',
            'dt': TIME_STEP,
            'tau': TAU,
            'x': X_VALS,
            'p': P_VALS,
            'ld_fwd': raster(rodeo.Direction.FORWARD),
            'ld_bwd': raster(rodeo.Direction.BACKWARD),
            'format': '[x][p]',
        }, fp)


def raster_ld(x_vals, p_vals, t, propagator, time_direction, max_time):
    ld = rodeo.LagrangianDescriptor(propagator,
        measure=rodeo.arc_length,
        time_direction=time_direction,
        max_time=max_time,
    )

    worker = functools.partial(ld, t=t)
    args = rodeo.grid(x_vals, p_vals).T

    with multiprocessing.Pool() as pool:
        return np.array(list(
            tqdm.tqdm(
                pool.imap(worker, args, chunksize=16),
                total=len(args), miniters=1,
            )
        )).reshape(len(x_vals), len(p_vals))


if __name__ == '__main__':
    sys.exit(int(main() or 0))
