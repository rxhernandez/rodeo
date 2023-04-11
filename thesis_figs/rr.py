#!/usr/bin/env python3

import functools
import multiprocessing
import pathlib
import sys

import numpy as np
import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'lib'))
import data
import rodeo


TIME = 0.0
TIME_STEP = 1e-3
X_VALS = np.linspace(-1.0, +1.0, 2048)
P_VALS = np.linspace(-1.0, +1.0, 2048)


class DoubleWell(rodeo.System):
    def force(self, qp, _t):
        return rodeo.position[qp] - rodeo.position[qp]**3


def main():
    propagator = rodeo.Propagator(DoubleWell(), rodeo.RungeKutta4, TIME_STEP)
    rr = raster_rr(X_VALS, P_VALS, TIME, propagator)

    data.save('generic/dwell/rr', {
        'system': 'V = x⁴/4 - x²/2',
        'dt': TIME_STEP,
        'x': X_VALS,
        'p': P_VALS,
        'rr': np.choose(rr, [0, 2, 3, 1]),
        'format': '[x][p]',
    })


def raster_rr(x_vals, p_vals, t, propagator):
    rr = rodeo.ReactiveRegion(propagator,
        reactant=rodeo.QLessEq(-1.0),
        product=rodeo.QGreaterEq(+1.0),
    )
    worker = functools.partial(rr, t=t)
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
