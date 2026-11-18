"""
Non-linearly matched pencil beam on the LHC horizontal primary collimator.

The pencil is generated at ``tcp.c6l7.b1`` with jaws at +/- 5 sigma from the
closed orbit, using a non-resonant normal form of the one-turn map at the
collimator. It is plotted in the horizontal phase space, in physical and in
Courant-Snyder normalised coordinates, together with orbits of the normal form
Hamiltonian, which the pencil follows, and the linear ellipses. With linear
coupling the jaw is not a single line in normalised coordinates: it is drawn
as a band of lines, one per vertical amplitude and phase.

Run from the repository root, with ``xnf4d`` installed (``pip install
".[examples]"``, see the README)::

    python examples/pencil_on_collimator.py
"""

import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt

from xnf4d.distributions import generate_pencil_on_collimator
from xnf4d.one_turn_map import OneTurnMap4D
from xnf4d.orbits import hamiltonian_orbits

LINE_FILE = "test_data/levelling.26_b1.json.gz"
COLLIMATOR = "tcp.c6l7.b1"
NEMITT_X = 3.5e-6
NEMITT_Y = 3.5e-6
JAW_SIGMAS = 5.0
SPREAD_SIGMAS = 0.1  # pencil spread, made large for visibility (xcoll default: 1 um)
NUM_PARTICLES = 10000
MAX_MAP_ORDER = 8
MAX_NF_ORDER = 7  # the normal form uses the map up to order MAX_NF_ORDER + 1
SIGMA_LIMIT = 8.0  # fit region of the one-turn map, beyond the jaw

# --------------------------------------------------------------------------- #
# Line
# --------------------------------------------------------------------------- #

line = xt.load(LINE_FILE)
line["dqx.b1_op"] = 0.0
line["dqy.b1_op"] = 0.0
line.build_tracker()
tw = line.twiss()

# --------------------------------------------------------------------------- #
# Normal form at the collimator, from a copy of the line starting there
# --------------------------------------------------------------------------- #

cycled = line.copy()
cycled.cycle(name_first_element=COLLIMATOR)
cycled.build_tracker()
otm = OneTurnMap4D(cycled, MAX_MAP_ORDER, NEMITT_X, NEMITT_Y, sigma_limit=SIGMA_LIMIT, seed=1)
otm.compute()
nf = otm.calculate_normal_form(MAX_NF_ORDER)  # non-resonant

# --------------------------------------------------------------------------- #
# Pencil on both jaws
# --------------------------------------------------------------------------- #

k = list(tw.name).index(COLLIMATOR)
gamma_beta = tw.particle_on_co.gamma0[0] * tw.particle_on_co.beta0[0]
sigma_x = np.sqrt(tw.betx[k] * NEMITT_X / gamma_beta)
jaw_L = tw.x[k] + JAW_SIGMAS * sigma_x
jaw_R = tw.x[k] - JAW_SIGMAS * sigma_x

particles = generate_pencil_on_collimator(
    line,
    COLLIMATOR,
    NUM_PARTICLES,
    normal_form=nf,
    plane="x",
    jaw=(jaw_L, jaw_R),
    nemitt_x=NEMITT_X,
    nemitt_y=NEMITT_Y,
    pencil_spread=SPREAD_SIGMAS * sigma_x,
    twiss=tw,
    seed=2,
)

# --------------------------------------------------------------------------- #
# Hamiltonian orbits and linear ellipses through points on the x_norm axis
# (y = py = 0), in normalised coordinates (x_norm, px_norm, y_norm, py_norm)
# --------------------------------------------------------------------------- #

# physical = co + W . normalised (transverse part, at the collimator)
W = tw.W_matrix[k][:4, :4] * np.sqrt(np.array([NEMITT_X, NEMITT_X, NEMITT_Y, NEMITT_Y]) / gamma_beta)
co = np.array([tw.x[k], tw.px[k], tw.y[k], tw.py[k]])
sigma_w = W[0, 0]  # x per unit of x_norm at y = py = 0


def to_physical(z):
    """Physical (x, px, y, py) of normalised coordinates of shape (4, N)."""
    return co[:, None] + W @ z


def orbits_norm(x_start):
    """Hamiltonian orbits through (x_start, px_norm=0, y=0, py=0)."""
    z0 = np.zeros((4, len(x_start)))
    z0[0] = (np.asarray(x_start) - co[0]) / sigma_w
    ho = hamiltonian_orbits(nf, z0, grid_size=801, extent=1.4 * np.max(np.abs(z0[0])))
    return [c for curves in ho.curves_norm for c in curves]


def ellipses_norm(x_start):
    """Linear ellipses (circles) through (x_start, px_norm=0, y=0, py=0)."""
    theta = np.linspace(0, 2 * np.pi, 721)
    circle = np.array([np.cos(theta), np.sin(theta), 0 * theta, 0 * theta])
    radii_norm = np.abs(np.asarray(x_start) - co[0]) / sigma_w
    return [r * circle for r in radii_norm]


radii = np.arange(1.0, JAW_SIGMAS + 2.0)
offsets = np.arange(-1.0, 3.01, 0.5) * SPREAD_SIGMAS * sigma_x
starts = {None: co[0] + radii * sigma_x}
starts.update({jaw: jaw + np.sign(jaw - co[0]) * offsets for jaw in (jaw_L, jaw_R)})
orbits = {key: orbits_norm(x0) for key, x0 in starts.items()}
ellipses = {key: ellipses_norm(x0) for key, x0 in starts.items()}

# --------------------------------------------------------------------------- #
# Plot in physical coordinates, where the jaws are vertical lines
# --------------------------------------------------------------------------- #

x, px = particles.x, particles.px
titles = {None: "Horizontal phase space", jaw_L: "Left jaw", jaw_R: "Right jaw"}
fig, axes = plt.subplots(1, 3, figsize=(18, 6), layout="constrained")
for ax, jaw in zip(axes, titles):
    for c in ellipses[jaw]:
        c = to_physical(c)
        ax.plot(c[0] * 1e3, c[1] * 1e6, color="tab:orange", ls="--", lw=1)
    for c in orbits[jaw]:
        c = to_physical(c)
        ax.plot(c[0] * 1e3, c[1] * 1e6, color="tab:red", lw=1)
    ax.plot(x * 1e3, px * 1e6, ".", color="tab:blue", ms=1.5)
    for j in (jaw_L, jaw_R):
        ax.axvline(j * 1e3, color="k", lw=1.5)
    if jaw is not None:
        sign = np.sign(jaw - co[0])
        on_side = np.sign(x - co[0]) == sign
        ax.set_xlim(*sorted(1e3 * (jaw + sign * np.array([-1, 3]) * SPREAD_SIGMAS * sigma_x)))
        pad = 0.3 * np.ptp(px[on_side])
        ax.set_ylim(1e6 * (px[on_side].min() - pad), 1e6 * (px[on_side].max() + pad))
    ax.set_xlabel(r"$x$ [mm]")
    ax.set_ylabel(r"$p_x$ [$\mu$rad]")
    ax.set_title(titles[jaw])
axes[0].plot([], [], color="tab:red", label="Hamiltonian orbits ($y = p_y = 0$)")
axes[0].plot([], [], color="tab:orange", ls="--", label="linear ellipses")
axes[0].plot([], [], ".", color="tab:blue", label="pencil")
axes[0].plot([], [], color="k", label=f"jaws ($\\pm${JAW_SIGMAS:g}$\\sigma$)")
axes[0].legend(loc="upper right")

# --------------------------------------------------------------------------- #
# Plot in Courant-Snyder normalised coordinates
# --------------------------------------------------------------------------- #

# With coupling, x depends on y_norm and py_norm, so the jaw x = jaw is a
# different line in (x_norm, px_norm) for each vertical amplitude r_y and
# phase phi_y:  W[0, :2] . (x_norm, px_norm) = jaw - x_co - W[0, 2:] . (y_norm, py_norm)
# For each r_y the lines fill a band; its two edges (the phases with the
# largest shift, +/- r_y |W[0, 2:]|) are drawn.
JAW_RY = np.arange(0.0, 3.01, 0.5)  # vertical amplitudes [sigma]
jaw_colors = plt.cm.Greys(np.linspace(0.9, 0.4, len(JAW_RY)))


def jaw_line(jaw, shift, pxn):
    """x_norm of the jaw line along px_norm, for W[0, 2:] . (y_norm, py_norm) = shift."""
    return (jaw - co[0] - shift - W[0, 1] * pxn) / W[0, 0]


def plot_jaw_band(ax, jaw, pxn=np.array([-50.0, 50.0])):
    """Jaw lines in (x_norm, px_norm) for the vertical amplitudes JAW_RY."""
    shift_max = JAW_RY.max() * np.hypot(*W[0, 2:])
    ax.fill_betweenx(pxn, jaw_line(jaw, -shift_max, pxn), jaw_line(jaw, shift_max, pxn),
                     color="0.9", lw=0, zorder=0)
    for r_y, color in zip(JAW_RY, jaw_colors):
        for shift in {-r_y * np.hypot(*W[0, 2:]), r_y * np.hypot(*W[0, 2:])}:
            ax.plot(jaw_line(jaw, shift, pxn), pxn, color=color, lw=1, zorder=1)


z = tw.get_normalized_coordinates(particles, nemitt_x=NEMITT_X, nemitt_y=NEMITT_Y)
xn, pxn = z.x_norm, z.px_norm
band = JAW_RY.max() * np.hypot(*W[0, 2:]) / W[0, 0]  # half-width of the band [sigma]
spread_n = SPREAD_SIGMAS * sigma_x / sigma_w

fig, axes = plt.subplots(1, 3, figsize=(18, 6), layout="constrained")
for ax, jaw in zip(axes, titles):
    for c in ellipses[jaw]:
        ax.plot(c[0], c[1], color="tab:orange", ls="--", lw=1)
    for c in orbits[jaw]:
        ax.plot(c[0], c[1], color="tab:red", lw=1)
    for j in (jaw_L, jaw_R):
        plot_jaw_band(ax, j)
    ax.plot(xn, pxn, ".", color="tab:blue", ms=1.5)
    if jaw is None:
        lim = JAW_SIGMAS + 2
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
    else:
        sign = np.sign(jaw - co[0])
        jaw_n = (jaw - co[0]) / sigma_w
        on_side = np.sign(xn) == sign
        ax.set_xlim(*sorted([jaw_n - sign * (band + 0.05), jaw_n + sign * (band + 3 * spread_n)]))
        pad = 0.3 * np.ptp(pxn[on_side])
        ax.set_ylim(pxn[on_side].min() - pad, pxn[on_side].max() + pad)
    ax.set_xlabel(r"$\hat{x}$ [$\sigma$]")
    ax.set_ylabel(r"$\hat{p}_x$ [$\sigma$]")
    ax.set_title(titles[jaw] + " (normalised)")
axes[0].plot([], [], color="tab:red", label="Hamiltonian orbits ($y = p_y = 0$)")
axes[0].plot([], [], color="tab:orange", ls="--", label="linear ellipses")
axes[0].plot([], [], ".", color="tab:blue", label="pencil")
for r_y, color in zip(JAW_RY[[0, -1]], jaw_colors[[0, -1]]):
    axes[0].plot([], [], color=color, label=f"jaw edges, $r_y = {r_y:g}\\sigma$")
axes[0].legend(loc="upper right")
plt.show()
