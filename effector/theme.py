"""effector's house theme: one designed, colorblind-safe palette plus a
matplotlib rcParams "chrome", ported from the vision reference palette.

Two layers, applied differently (see LOGBOOK #25):

* **palette colors** — semantic tokens (``MEAN``, ``BAND``, ``CLOUD``,
  ``BAR_FACE`` …) are baked into every draw call in ``visualization.py``. They
  always apply, per-artist, and never touch matplotlib's global state.
* **chrome rcParams** — background, font, grid, spines, prop-cycle. These mutate
  matplotlib's *global* ``rcParams`` and therefore only take effect when the user
  opts in via :func:`set_theme`. (matplotlib re-reads grid/tick rcParams at draw
  time — after a plot returns ``(fig, ax)`` — so a local ``rc_context`` would
  revert at render; a persistent global update is the only thing that survives.)

``set_theme("light" | "dark" | "paper")`` switches the active theme (colors and
chrome together); ``set_theme("default")`` restores stock matplotlib.
"""

import dataclasses

import matplotlib as mpl
from cycler import cycler

# --- validated categorical palette (light column) ------------------------------
BLUE = "#2a78d6"
AQUA = "#1baf7a"
YELLOW = "#eda100"
GREEN = "#008300"
VIOLET = "#4a3aa7"
RED = "#e34948"
MAGENTA = "#e87ba4"
ORANGE = "#eb6834"
CAT = (BLUE, AQUA, YELLOW, GREEN, VIOLET, RED, MAGENTA, ORANGE)

# sequential blue ramp (100 -> 700)
SEQ = (
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
)

# chrome & ink (light)
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# chrome & ink (dark)
D_SURFACE = "#1b1b1a"
D_INK = "#f2f2ef"
D_INK2 = "#c9c8c4"
D_MUTED = "#9a9992"
D_GRID = "#33322f"
D_BASELINE = "#4a4946"

# status (annotations only — carried for future use)
GOOD = "#0ca30c"
WARNING = "#fab219"
SERIOUS = "#ec835a"
CRITICAL = "#d03b3b"
DIV_MID = "#f0efec"


def _light_rcparams():
    """The light chrome — the ported ``use_style()`` dict plus the CAT prop-cycle."""
    return {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 9,
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 2.0,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
        "axes.prop_cycle": cycler(color=list(CAT)),
    }


@dataclasses.dataclass(frozen=True)
class Theme:
    """A named look: semantic color tokens + the rcParams chrome. Tokens are read
    at draw time via :func:`active`; rcParams are pushed by :func:`set_theme`."""

    name: str
    rcparams: dict
    # semantic color tokens (what the draw calls reference)
    MEAN: str  # mean-effect line
    BAND: str  # std / std_err heterogeneity fill
    CLOUD: str  # ICE / SHAP point clouds
    BAR_FACE: str  # categorical bar face
    BAR_EDGE: str  # categorical bar edge
    BAR_FACE_MUTED: tuple  # (RH)ALE bin bar face (near-transparent RGBA)
    BAR_EDGE_ACCENT: str  # (RH)ALE bin bar edge
    ERROR: str  # error-bar / whisker color
    AVG: str  # average-output baseline
    CONNECT: str  # (RH)ALE accumulation line
    CAT: tuple  # per-series cycle for method comparison
    # alpha / width knobs (the taste dials tuned during figure review)
    BAND_ALPHA: float = 0.25
    CLOUD_ALPHA: float = 0.12
    CLOUD_LW: float = 0.8
    DOT_ALPHA: float = 0.35
    SHAP_MARKER_ALPHA: float = 0.5


LIGHT = Theme(
    name="light",
    rcparams=_light_rcparams(),
    MEAN=BLUE,
    BAND=MUTED,
    CLOUD=MUTED,
    BAR_FACE=BLUE,
    BAR_EDGE=INK,
    BAR_FACE_MUTED=(0.1, 0.1, 0.1, 0.1),
    BAR_EDGE_ACCENT=BLUE,
    ERROR=INK2,
    AVG=INK,
    CONNECT="#104281",  # SEQ dark blue
    CAT=CAT,
)

DARK = dataclasses.replace(
    LIGHT,
    name="dark",
    rcparams={
        **_light_rcparams(),
        "figure.facecolor": D_SURFACE,
        "axes.facecolor": D_SURFACE,
        "savefig.facecolor": D_SURFACE,
        "axes.edgecolor": D_BASELINE,
        "axes.labelcolor": D_INK2,
        "axes.titlecolor": D_INK,
        "grid.color": D_GRID,
        "xtick.color": D_MUTED,
        "ytick.color": D_MUTED,
    },
    BAND=D_MUTED,
    CLOUD=D_MUTED,
    BAR_EDGE=D_INK,
    AVG=D_INK,
    # MEAN/BAR_FACE stay palette blue (reads on dark); CAT unchanged
)

PAPER = dataclasses.replace(
    LIGHT,
    name="paper",
    rcparams={
        **_light_rcparams(),
        "figure.facecolor": "#ffffff",
        "axes.facecolor": "#ffffff",
        "savefig.facecolor": "#ffffff",
        "axes.grid": False,
        "savefig.dpi": 300,
    },
)

THEMES = {"light": LIGHT, "dark": DARK, "paper": PAPER}

_ACTIVE = LIGHT


def active():
    """The currently active :class:`Theme` (light by default). Draw calls read
    their color tokens from here, so switching themes recolors new figures."""
    return _ACTIVE


def apply_theme(theme):
    """Push a theme's chrome into matplotlib's global ``rcParams``."""
    mpl.rcParams.update(theme.rcparams)


def set_theme(name="light"):
    """Activate a house theme for every figure effector draws after this call.

    ```python
    effector.set_theme("dark")
    effector.set_theme("default")   # back to stock matplotlib
    ```

    | Name | Look |
    |---|---|
    | `"light"` | the default: off-white surface, colorblind-safe palette |
    | `"dark"` | the same palette on a dark surface |
    | `"paper"` | pure white, no grid, 300-dpi savefig — print-ready |
    | `"default"` | restore stock matplotlib rcParams |

    !!! note "Global by design"
        The chrome (background, font, grid, spines) lives in matplotlib's
        global ``rcParams`` — matplotlib re-reads those at draw time, so a
        local ``rc_context`` would revert before render. Palette colors and
        chrome switch together for *new* figures; existing figures keep their
        look.

    Args:
        name: one of the names above (default `"light"`).

    Raises:
        ValueError: unknown theme name.
    """
    global _ACTIVE
    if name == "default":
        mpl.rcParams.update(mpl.rcParamsDefault)
        _ACTIVE = LIGHT
        return
    if name not in THEMES:
        valid = sorted(THEMES) + ["default"]
        raise ValueError(f"unknown theme {name!r}; choose from {valid}")
    _ACTIVE = THEMES[name]
    apply_theme(_ACTIVE)
