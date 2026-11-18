"""
Non-linearly matched Gaussian bunch at the LHC horizontal primary collimator.

A Gaussian bunch is generated at ``tcp.c6l7.b1``, using a non-resonant normal
form of the one-turn map there; the collimator is only used as the place
where the distribution is generated. The transverse coordinates are Gaussian
in normal form coordinates, the longitudinal ones are matched to the bucket,
as in ``xpart``.

The horizontal and vertical phase spaces are plotted as 2D histograms in
Courant-Snyder normalised and in physical coordinates, with orbits of the normal form
Hamiltonian and the linear orbits (circles) through the same points; the
longitudinal phase space is plotted without orbits.

Run from the repository root, with ``xnf4d`` installed (``pip install
".[examples]"``, see the README)::

    python examples/gaussian_bunch_at_collimator.py
"""

import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt
from matplotlib.colors import LogNorm

from xnf4d.distributions import generate_matched_gaussian_bunch
from xnf4d.one_turn_map import OneTurnMap4D
from xnf4d.orbits import hamiltonian_orbits

LINE_FILE = "test_data/levelling.26_b1.json.gz"
ELEMENT = "tcp.c6l7.b1"
NEMITT_X = 3.5e-6
NEMITT_Y = 3.5e-6
SIGMA_Z = 7.61e-2
NUM_PARTICLES = 200_000
MAX_MAP_ORDER = 8
MAX_NF_ORDER = 7  # the normal form uses the map up to order MAX_NF_ORDER + 1
SIGMA_LIMIT = 8.0  # fit region of the one-turn map, covering the Gaussian tails

# --------------------------------------------------------------------------- #
# Line
# --------------------------------------------------------------------------- #

line = xt.load(LINE_FILE)
line["dqx.b1_op"] = 0.0
line["dqy.b1_op"] = 0.0
line.build_tracker()
tw = line.twiss()

# --------------------------------------------------------------------------- #
# Normal form at the element, from a copy of the line starting there
# --------------------------------------------------------------------------- #

cycled = line.copy()
cycled.cycle(name_first_element=ELEMENT)
cycled.build_tracker()
otm = OneTurnMap4D(cycled, MAX_MAP_ORDER, NEMITT_X, NEMITT_Y, sigma_limit=SIGMA_LIMIT, seed=1)
otm.compute()
nf = otm.calculate_normal_form(MAX_NF_ORDER)  # non-resonant

# --------------------------------------------------------------------------- #
# Gaussian bunch
# --------------------------------------------------------------------------- #

particles = generate_matched_gaussian_bunch(
    NUM_PARTICLES, NEMITT_X, NEMITT_Y, SIGMA_Z,
    line=line, normal_form=nf, at_element=ELEMENT, twiss=tw, seed=2,
)

# --------------------------------------------------------------------------- #
# Orbits through (r, 0) in the normalised phase space of each plane, with the
# other plane at zero: Hamiltonian orbits and linear orbits (circles)
# --------------------------------------------------------------------------- #

radii = np.arange(1.0, 6.0)


def nonlinear_orbits(plane):
    """Hamiltonian orbits through (r, 0) in `plane`, for r in `radii`."""
    start = np.zeros((4, len(radii)))
    start[0 if plane == "x" else 2] = radii
    ho = hamiltonian_orbits(nf, start, grid_size=801, extent=1.3 * radii.max())
    return [c for curves in ho.curves_norm for c in curves]


theta_c = np.linspace(0, 2 * np.pi, 361)
orbits = {"x": nonlinear_orbits("x"), "y": nonlinear_orbits("y")}

# --------------------------------------------------------------------------- #
# Plot, in normalised and in physical coordinates
# --------------------------------------------------------------------------- #

# physical = closed orbit + W . normalised (transverse part, at the element)
k = list(tw.name).index(ELEMENT)
gamma_beta = tw.particle_on_co.gamma0[0] * tw.particle_on_co.beta0[0]
W = tw.W_matrix[k][:4, :4] * np.sqrt(np.array([NEMITT_X, NEMITT_X, NEMITT_Y, NEMITT_Y]) / gamma_beta)
co = np.array([tw.x[k], tw.px[k], tw.y[k], tw.py[k]])
units = np.array([1e3, 1e6, 1e3, 1e6])[:, None]  # mm, urad
norm = tw.get_normalized_coordinates(particles, nemitt_x=NEMITT_X, nemitt_y=NEMITT_Y)
lim = radii.max() + 0.5


def circle(radius, plane):
    """Linear orbit of normalised radius `radius` in `plane`, shape (4, N)."""
    c = np.zeros((4, len(theta_c)))
    i = 0 if plane == "x" else 2
    c[i], c[i + 1] = radius * np.cos(theta_c), radius * np.sin(theta_c)
    return c


def plot_figure(physical):
    """Horizontal, vertical and longitudinal phase space."""
    # in physical coordinates the orbits are those of on-momentum particles:
    # the bunch is smeared horizontally by the dispersion (D_x sigma_delta)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), layout="constrained")
    for ax, plane in zip(axes, ("x", "y")):
        i = 0 if plane == "x" else 2
        if physical:
            transform = lambda c: (co[:, None] + W @ c) * units
            u, pu = getattr(particles, plane) * 1e3, getattr(particles, "p" + plane) * 1e6
            # bins covering the linear orbit of radius lim
            outer = transform(circle(lim, plane))
            edges = [np.linspace(outer[j].min(), outer[j].max(), 151) for j in (i, i + 1)]
            ax.set_xlabel(rf"${plane}$ [mm]")
            ax.set_ylabel(rf"$p_{plane}$ [$\mu$rad]")
        else:
            transform = lambda c: c
            u, pu = getattr(norm, plane + "_norm"), getattr(norm, "p" + plane + "_norm")
            edges = [np.linspace(-lim, lim, 151)] * 2
            ax.set_aspect("equal")
            ax.set_xlabel(rf"$\hat{{{plane}}}$ [$\sigma$]")
            ax.set_ylabel(rf"$\hat{{p}}_{plane}$ [$\sigma$]")
        *_, image = ax.hist2d(u, pu, bins=edges, cmap="Blues", norm=LogNorm())
        for rr in radii:
            c = transform(circle(rr, plane))
            ax.plot(c[i], c[i + 1], color="tab:orange", ls="--", lw=1)
        for c in orbits[plane]:
            c = transform(c)
            ax.plot(c[i], c[i + 1], color="tab:red", lw=1)
        ax.set_title("Horizontal phase space" if plane == "x" else "Vertical phase space")
        fig.colorbar(image, ax=ax, label="particles per bin", shrink=0.8)
    axes[0].plot([], [], color="tab:red", label="Hamiltonian orbits")
    axes[0].plot([], [], color="tab:orange", ls="--", label="linear orbits")
    axes[0].legend(loc="upper right")

    ax = axes[2]
    *_, image = ax.hist2d(particles.zeta * 1e3, particles.delta * 1e4, bins=150, cmap="Blues", norm=LogNorm())
    ax.set_xlabel(r"$\zeta$ [mm]")
    ax.set_ylabel(r"$\delta$ [$10^{-4}$]")
    ax.set_title("Longitudinal phase space")
    fig.colorbar(image, ax=ax, label="particles per bin", shrink=0.8)
    fig.suptitle("Physical coordinates" if physical else "Normalised coordinates")


plot_figure(physical=False)
plot_figure(physical=True)
plt.show()
