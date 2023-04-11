import pathlib

import matplotlib
matplotlib.use('pgf')

import matplotlib.colors as mcolors
import matplotlib.figure as mfigure
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
import numpy as np


GOLDEN = (1 + np.sqrt(5)) / 2
PT = 1 / 72.27
BP = 1 / 72
TEXT_WIDTH = 426.79141 * PT
CBAR_WIDTH = 10 * BP
DEFAULT_DPI = 300
DEFAULT_CMAP = plt.get_cmap('inferno')
REGIONS_CMAP = mcolors.ListedColormap((
    'lightblue', 'lightgreen', 'tab:green', 'tab:blue',
), 'rr')
REGIONS_LABELS = (
    r'$\mathrm{R} \to \mathrm{R}$', r'$\mathrm{P} \to \mathrm{R}$',
    r'$\mathrm{R} \to \mathrm{P}$', r'$\mathrm{P} \to \mathrm{P}$',
)
REGIONS_TICKS = tuple(range(len(REGIONS_LABELS)))


def log(msg, *args, **kwargs):
    print('  ' + msg.format(*args, **kwargs))


def fmt_num(num, fmt='g'):
    return rf'{num:{fmt}}'


def newfig(
    width=1.0, aspect=GOLDEN, nrows=1, ncols=1, gridspec=False,
    left=1, right=1, top=1, bottom=1, wspace=6, hspace=6,
    **kwargs,
):
    assert 'gridspec_kw' not in kwargs
    kwargs.setdefault('dpi', DEFAULT_DPI)

    width *= TEXT_WIDTH
    left, right, top, bottom = left * BP, right * BP, top * BP, bottom * BP
    wspace, hspace = wspace * BP, hspace * BP

    axes_width = (width - left - right - wspace * (ncols - 1)) / ncols
    axes_height = axes_width / aspect
    height = axes_height * nrows + top + bottom + hspace * (nrows - 1)

    if gridspec:
        gs_kwargs = {
            name: kwargs.pop(name, None)
            for name in ('width_ratios', 'height_ratios')
        }
        fig = plt.figure(figsize=(width, height), **kwargs)
        axes_or_gs = fig.add_gridspec(nrows, ncols, **gs_kwargs)
    else:
        fig, axes_or_gs = plt.subplots(
            figsize=(width, height), nrows=nrows, ncols=ncols, **kwargs)

    fig.subplots_adjust(
        left=left / width, right=1 - right / width,
        top=1 - top / height, bottom=bottom / height,
        wspace=wspace / axes_width, hspace=hspace / axes_height,
    )

    return fig, axes_or_gs


def save_figure(fig, path, close=True):
    log('Writing figure to PDF...')
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path))
    if close:
        plt.close(fig)


def extent(data, x='x', y='y'):
    return tuple(data[x].reshape(-1)[[0, -1]]) + tuple(data[y].reshape(-1)[[0, -1]])


def fig_hspace(ax):
    sp = ax.figure.subplotpars
    gs = ax.get_gridspec()
    return sp.hspace * (sp.top - sp.bottom) / (gs.nrows + sp.hspace * (gs.nrows - 1))


def subfig_label(ax, idx, ha, x, dx, va, y, dy, **kwargs):
    label = chr(ord('a') + idx) if isinstance(idx, int) else str(idx)
    text(ax, ha, x, dx, va, y, dy, rf'\textbf{{({label})}}', **kwargs)


def imshow(ax, data, extent, cmap=DEFAULT_CMAP, interp=True, **kwargs):
    interp = 'spline16' if interp is True else None if not interp else interp
    return ax.imshow(
        data,
        cmap=cmap,
        aspect='auto',
        interpolation=interp,
        origin='lower',
        extent=extent,
        **kwargs,
    )


def regions(ax, data, extent, cmap=REGIONS_CMAP):
    return imshow(ax, data, extent, cmap,
        interp=False, vmin=-0.5, vmax=cmap.N - 0.5)


def cbar_above(fig, axes, aximg, dy=None, **kwargs):
    pos = [axes[idx].get_position() for idx in (0, -1)]
    dy = fig_hspace(axes[0]) if dy is None else dy
    cax = fig.add_axes((
        pos[0].xmin              , pos[0].ymax + dy                ,
        pos[1].xmax - pos[0].xmin, CBAR_WIDTH / fig.get_figheight(),
    ))
    cbar = fig.colorbar(aximg, cax=cax, orientation='horizontal', **kwargs)
    cax.xaxis.set_ticks_position('top')
    cax.xaxis.set_label_position('top')
    return cbar, cax


def cbar_minmax_labels(cbar, labels=None, fmt='g'):
    assert cbar.orientation == 'horizontal'
    labels = labels or [fmt_num(x, fmt) for x in cbar.mappable.get_clim()]
    cbar.set_ticks(cbar.mappable.get_clim(), labels=labels)
    for align, label in zip(('left', 'right'), cbar.ax.xaxis.get_ticklabels()):
        label.set_horizontalalignment(align)


def text(ax, ha, x, dx, va, y, dy, txt, **kwargs):
    trans = ax.transAxes + mtrans.ScaledTranslation(
        dx * BP, dy * BP, ax.figure.dpi_scale_trans)
    ax.text(x, y, txt, transform=trans, ha=ha, va=va, **kwargs)


def _init_matplotlib():
    plt.rc('font', family='serif', size=12)
    plt.rc('text', usetex=True)
    plt.rc('pgf', rcfonts=False, preamble=rf"""
        %\usepackage{{xcolor}}
    """)


_init_matplotlib()
