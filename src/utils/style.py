"""Shared figure style: widths, font sizes and semantic colours.

Figure scripts call apply_paper_style() once in main(); importing has no side effects.
"""

from typing import Union
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt


# Widths ----------------------------------------------------------------------

TEXT_WIDTH = 5.5              # LaTeX \linewidth, inches

# Font sizes ------------------------------------------------------------------

TICK_LABEL_SIZE  = 6
AXIS_LABEL_SIZE  = 8
PANEL_TITLE_SIZE = 8          # descriptive titles ("Point attractor")
PANEL_LABEL_SIZE = 9          # the bold (a)/(b) letters
LEGEND_SIZE      = 6
MATH_LABEL_SIZE  = 10         # CM mathtext looks smaller than the serif text

# Fonts -----------------------------------------------------------------------

SERIF_STACK = ["Nimbus Roman", "Times New Roman", "Times", "DejaVu Serif"]

# Colours: model class --------------------------------------------------------

LINEAR_COLOR    = "#1d91c0"   # muted blue
NONLINEAR_COLOR = "#ae017e"   # magenta-red

# Colours: ground truth in flow-field panels ----------------------------------

TRUE_FLOW_COLOR = "0.55"      # grey arrows
TRUE_TRAJ_COLOR = "black"

# Colours: stim-timing factor (Okabe-Ito) -------------------------------------

TIMING_COLOR_REGULAR = "#E69F00"   # orange
TIMING_COLOR_POISSON = "#009E73"   # teal-green

# Colours: intervention-hardness models (Okabe-Ito) ---------------------------

HARDNESS_MODEL_COLORS = {
    "m1": NONLINEAR_COLOR,   # iSSM
    "m2": "#41b6c4",         # aquamarine
    "m3": "#E69F00",         # orange
}

# Colours: seizure zone -------------------------------------------------------

ZONE_HEALTHY_COLOR = "0.55"      # grey
ZONE_EPI_COLOR     = "#E69F00"   # Okabe-Ito orange

# Colours: pass/fail marks under qualitative summary panels -------------------

VERDICT_OK_COLOR  = "#2ca02c"   # green tick
VERDICT_BAD_COLOR = "#d62728"   # red cross

# Nimbus Roman has no dingbats, so the glyphs use DejaVu Sans (ships with matplotlib).
VERDICT_MARKS = {True: "\u2714", False: "\u2718"}
VERDICT_FONT  = "DejaVu Sans"

# Non-colour drawing constants ------------------------------------------------

STIM_ALPHA = 0.5

# A +/-std band may widen a panel's y-range by at most this factor; see std_clipped_ylim.
STD_RANGE_FACTOR = 1.25


# Style application ------------------------------------------------------------

def apply_paper_style() -> None:
    """Apply the figure rcParams; call once at the top of a figure script's main()."""
    plt.rcParams.update({
        "font.family":      "serif",
        "font.serif":       SERIF_STACK,
        # Maths stays Computer Modern.
        "mathtext.fontset": "cm",
        # Used only by the "custom" fontset (panel_label). cal is set because the
        # default "cursive" is usually missing and would warn on every call.
        "mathtext.rm":      "Nimbus Roman",
        "mathtext.bf":      "Nimbus Roman:bold",
        "mathtext.it":      "Nimbus Roman:italic",
        "mathtext.cal":     "Nimbus Roman:italic",
        "pdf.fonttype":     42,   # TrueType, not Type 3
        "ps.fonttype":      42,
        "xtick.labelsize":  TICK_LABEL_SIZE,
        "ytick.labelsize":  TICK_LABEL_SIZE,
        "axes.labelsize":   AXIS_LABEL_SIZE,
        "axes.titlesize":   PANEL_TITLE_SIZE,
        "legend.fontsize":  LEGEND_SIZE,
    })


def panel_label(
    target: Union[plt.Axes, plt.Figure],
    letter: str,
    x: float = -0.18,
    y: float = 1.08,
    bold: bool = True,
    size: float = PANEL_LABEL_SIZE,
    **kwargs,
) -> "mpl.text.Text":
    """Place a panel label "(a)" on an Axes (axes coords) or Figure (figure coords).

    With bold=True only the letter is bold; matplotlib cannot mix weights in one Text,
    so this uses mathtext with the "custom" fontset set up by apply_paper_style().
    """
    if bold:
        text, extra = rf"$(\mathbf{{{letter}}})$", {"math_fontfamily": "custom"}
    else:
        text, extra = f"({letter})", {}

    kwargs = {"ha": "left", "va": "baseline", "fontsize": size, **extra, **kwargs}
    if isinstance(target, plt.Axes):
        return target.text(x, y, text, transform=target.transAxes, **kwargs)
    return target.text(x, y, text, **kwargs)


def save_figure(fig: plt.Figure, path, fmt: str = "pdf", dpi: int = 300) -> None:
    """Save without bbox_inches="tight", so the file keeps its declared width and font sizes."""
    path = str(path)
    if not path.endswith(f".{fmt}"):
        path = f"{path}.{fmt}"
    fig.savefig(path, format=fmt, dpi=dpi)


def fit_to_ink(fig: plt.Figure, axes: list, margin: float, dpi: int = 300) -> None:
    """Centre the ink horizontally and crop the height to it plus margin (inches).

    Measured on rendered pixels, since text bboxes include font padding. Only the
    given axes are moved, so any legend must be attached to one of them.
    """
    fig.set_dpi(dpi)
    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    rows, cols = np.nonzero((rgb < 250).any(axis=2))
    w, h = fig.get_size_inches()
    x0, x1 = cols.min() / dpi, (cols.max() + 1) / dpi
    y0, y1 = h - (rows.max() + 1) / dpi, h - rows.min() / dpi
    dx = (w - x1 - x0) / 2
    dy = margin - y0
    new_h = y1 - y0 + 2 * margin
    for ax in axes:
        p = ax.get_position()
        ax.set_position([p.x0 + dx / w, (p.y0 * h + dy) / new_h,
                         p.width, p.height * h / new_h])
    fig.set_size_inches(w, new_h)
