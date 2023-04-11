import matplotlib
matplotlib.use('pgf')

import matplotlib.colors as mcolors
import matplotlib.figure as mfigure
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
import numpy as np

import color as _color
import data as _data
import paths as _paths


GOLDEN = (1 + np.sqrt(5)) / 2
PT = 1 / 72.27
BP = 1 / 72
TEXT_WIDTH = 426.79141 * PT
CBAR_WIDTH = 10 * BP
DEFAULT_DPI = 300
DEFAULT_CMAP = plt.get_cmap('inferno')
REGIONS_CMAP = mcolors.ListedColormap((
    'jr:light_blue', 'jr:light_green', 'jr:green', 'jr:blue',
), 'jr:rr')
REGIONS_LABELS = (r'\RR{R}{R}', r'\RR{P}{R}', r'\RR{R}{P}', r'\RR{P}{P}')
REGIONS_TICKS = tuple(range(len(REGIONS_LABELS)))


def log(msg, *args, **kwargs):
    print('  ' + msg.format(*args, **kwargs))


def load_data(name):
    return _data.load(f'{_paths.model_system()}/{name}')


def find_data(base, pattern='.*'):
    return _data.find(f'{_paths.model_system()}/{base}', pattern)


def fmt_num(num, fmt='g'):
    return rf'\num{{{num:{fmt}}}}'


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


def save_figure(fig, suffix='', close=True):
    log('Writing figure to PDF...')

    path = _paths.figure_path(suffix)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(str(path))
    if close:
        plt.close(fig)


def extent(data, x='x', y='y'):
    return tuple(data[x].reshape(-1)[[0, -1]]) + tuple(data[y].reshape(-1)[[0, -1]])


def fig_wspace(ax):
    sp = ax.figure.subplotpars
    gs = ax.get_gridspec()
    return sp.wspace * (sp.right - sp.left) / (gs.ncols + sp.wspace * (gs.ncols - 1))


def fig_hspace(ax):
    sp = ax.figure.subplotpars
    gs = ax.get_gridspec()
    return sp.hspace * (sp.top - sp.bottom) / (gs.nrows + sp.hspace * (gs.nrows - 1))


def subfig_label(ax, idx, ha, x, dx, va, y, dy, **kwargs):
    label = chr(ord('a') + idx) if isinstance(idx, int) else str(idx)
    text(ax, ha, x, dx, va, y, dy, rf'\textbf{{({label})}}', **kwargs)


def auto_xlim_aspect_1(ax, offset=0.0):
    y_min, y_max = ax.get_ylim()
    width, height = np.abs(ax.get_window_extent().size)
    dx = width / height * (y_max - y_min)
    ax.set_xlim(np.array([-0.5, +0.5]) * dx + offset)


def common_lims(axis, axes, vmin=None, vmax=None):
    clims = np.array([getattr(ax, f'get_{axis}lim')() for ax in axes])
    vmin = clims.min() if vmin is None else vmin
    vmax = clims.max() if vmax is None else vmax

    for ax in axes:
        getattr(ax, f'set_{axis}lim')(vmin, vmax)


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


def psos(
    ax, data, x, y,
    center=(0.0, 0.0), color=('jr:blue', 'jr:light_blue'), shades=4, ms=2,
    **kwargs,
):
    data_2d = [traj[~np.isnan(traj).any(axis=-1)][:, [x, y]] for traj in data]
    sq_radii = [np.sum((traj - center)**2, axis=-1) for traj in data_2d]
    sort_idcs = np.argsort([np.amin(r2) for r2 in sq_radii])

    if isinstance(color, str) or len(color) != 2:
        color = (color, color)

    colors = _color.cycle(*color, shades, bidir=True)
    for idx, c in zip(sort_idcs, colors):
        ax.plot(*data_2d[idx].T, color=c, ls='none', marker='o', ms=ms, **kwargs)


def markers(
    ax, x, y, *args,
    marker='o', ms=8, mec='jr:white', mew=0.5, ls='none', **kwargs,
):
    ax.plot(x, y, *args, marker=marker, ms=ms, mec=mec, mew=mew, ls=ls, **kwargs)


def cbar_beside(fig, axes, aximg, dx=None, **kwargs):
    pos = [axes[idx].get_position() for idx in (0, -1)]
    dx = fig_wspace(axes[0]) if dx is None else dx
    cax = fig.add_axes((
        pos[0].xmax + dx               , pos[1].ymin              ,
        CBAR_WIDTH / fig.get_figwidth(), pos[0].ymax - pos[1].ymin,
    ))
    cbar = fig.colorbar(aximg, cax=cax, orientation='vertical', **kwargs)
    return cbar, cax


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


def legend(ax, *args, **kwargs):
    opts = dict(
        columnspacing=1.5,
        handlelength=1.5,
        handletextpad=0.5,
        borderaxespad=0.04,
        edgecolor='none',
        frameon=False,
    )

    return ax.legend(*args, **(opts | kwargs))


def text(ax, ha, x, dx, va, y, dy, txt, **kwargs):
    trans = ax.transAxes + mtrans.ScaledTranslation(
        dx * BP, dy * BP, ax.figure.dpi_scale_trans)
    ax.text(x, y, txt, transform=trans, ha=ha, va=va, **kwargs)


def arrow(fig_or_ax, from_pos, to_pos, **kwargs):
    kwargs.setdefault('color', 'jr:black')
    kwargs.setdefault('lw', 1.5)
    kwargs.setdefault('arrowstyle', 'fancy, head_width=6, head_length=6, tail_width=1e-12')
    if isinstance(fig_or_ax, mfigure.FigureBase):
        kwargs.setdefault('transform', fig_or_ax.transFigure)

    fig_or_ax.add_artist(mpatches.FancyArrowPatch(from_pos, to_pos, **kwargs))


def _init_matplotlib():
    plt.rc('font', family='serif', size=12)
    plt.rc('text', usetex=True)
    plt.rc('pgf', rcfonts=False, texsystem='lualatex', preamble=rf"""
        \usepackage{{jr-fonts}}
        \usepackage{{xcolor}}
        \selectcolormodel{{rgb}}
        \input{{{_paths.TEX_PREAMBLE}}}
        \frenchspacing
        \let\displaystyle\textstyle
    """)

    plt.rc('contour', linewidth=1)

    _color.set_colors()
    _paths.set_texinputs()


_init_matplotlib()
