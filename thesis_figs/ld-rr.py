#!/usr/bin/env python3

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'lib'))
import plot


LD_CUT_STYLE = dict(color='jr:gray', lw=1)
RR_CUT_STYLE = dict(color='jr:dark_gray', lw=1)
CIRCLE_BG = dict(bbox={'boxstyle': 'circle', 'facecolor': 'jr:white', 'fill': True})

CUT_X = -0.8

RR_LABELS = (
    ('center', -0.5, 'center'  ,  0.0, r'\num{1}', CIRCLE_BG),
    ('center', +0.5, 'center'  ,  0.0, r'\num{2}', CIRCLE_BG),
    ('center',  0.0, 'center'  , +0.5, r'\num{3}', CIRCLE_BG),
    ('center',  0.0, 'center'  , -0.5, r'\num{4}', CIRCLE_BG),
)


def main():
    ld_data, rr_data = load_data()

    fig, axes = plot.newfig(aspect=4 / 3, nrows=2, ncols=2, height_ratios=(2, 1),
        left=47, right=9, top=44, bottom=34, wspace=53, hspace=36)

    for ax in axes[0]:
        ax.set_xlabel('$x$')
        ax.set_ylabel('$p_x$', labelpad=0)
    for ax in axes[1]:
        ax.set_xlabel('$p_x$')
        ax.margins(x=0)

    for idx, ax in enumerate(axes[0]):
        plot.subfig_label(ax, idx, 'left', 0, -47, 'top', 1, 43)
    for idx, ax in enumerate(axes[1]):
        plot.subfig_label(ax, idx + 2, 'left', 0, -47, 'top', 1, 0)

    plot_ld(*axes.T[0], ld_data)
    plot_rr(*axes.T[1], rr_data)

    plot.save_figure(fig)


def load_data():
    yield plot.load_data('dwell/ld')
    yield plot.load_data('dwell/rr')
    plot.log('Data loaded')


def plot_ld(main_ax, cut_ax, data):
    ld = data['ld_bwd'] + data['ld_fwd']
    cut_x, cut_ld = cut(ld, data['x'], CUT_X)

    aximg = plot.imshow(main_ax, ld.T, plot.extent(data, 'x', 'p'), vmax=13)
    main_ax.axvline(cut_x, **LD_CUT_STYLE)

    fig = main_ax.figure
    cbar, _ = plot.cbar_above(fig, [main_ax, main_ax], aximg,
        dy=h_bp(fig, 6), extend='max')
    plot.cbar_minmax_labels(cbar, ['min', 'max'])
    cbar.set_label(r'$\LD$', labelpad=-4)

    cut_ax.set_ylabel(r'$\LD$', labelpad=14)
    cut_ax.plot(data['p'], cut_ld)


def plot_rr(main_ax, cut_ax, data):
    cut_x, cut_rr = cut(data['rr'], data['x'], CUT_X)

    aximg = plot.regions(main_ax, data['rr'].T, plot.extent(data, 'x', 'p'))
    main_ax.axvline(cut_x, **RR_CUT_STYLE)
    for ha, x, va, y, text, kwargs in RR_LABELS:
        main_ax.text(x, y, text, ha=ha, va=va, **kwargs)

    fig = main_ax.figure
    cbar, cax = plot.cbar_above(fig, [main_ax, main_ax], aximg, dy=h_bp(fig, 6))
    cbar.set_ticks(plot.REGIONS_TICKS, labels=plot.REGIONS_LABELS, rotation=30)
    cax.tick_params(pad=-2)

    cut_ax.set_ylabel('region', labelpad=18)
    cut_ax.set_yticks([1, 2, 3, 4])
    cut_ax.plot(data['p'], np.choose(cut_rr, [1, 4, 3, 2]))


def h_bp(fig, pt):
    return pt * plot.BP / fig.get_figheight()


def cut(array, xvals, x):
    idx = np.abs(xvals - x).argmin()
    return xvals[idx], array[idx]


if __name__ == '__main__':
    sys.exit(int(main() or 0))
