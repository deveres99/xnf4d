"""
Non-linearly matched Gaussian and q-Gaussian bunches at the LHC horizontal
primary collimator.

Five bunches are generated at ``tcp.c6l7.b1``, using a non-resonant normal
form of the one-turn map there (the collimator is only used as the place
where the distributions are generated):

1. a Gaussian bunch;
2. - 5. q-Gaussian bunches (``q = 1.5``, ``beta = 1.68``), factorisable or
   not, cut at 5 sigma in both planes with a skew (elliptical) or a
   rectangular cut on the normalised radii ``r_x``, ``r_y``.

For each bunch, 2D histograms of ``(x_norm, px_norm)`` and ``(y_norm, py_norm)``
(with the orbits of the normal form Hamiltonian and the linear orbits through
the same points), ``(x_norm, y_norm)``, ``(px_norm, py_norm)`` and
``(zeta, delta)`` are plotted, and the distributions of ``x_norm`` of all
bunches are compared.

Run from the repository root, with ``xnf4d`` installed (``pip install
".[examples]"``, see the README)::

    python examples/qgaussian_bunches_at_collimator.py
"""

import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt
from matplotlib.colors import LogNorm

from xnf4d.distributions import generate_matched_gaussian_bunch, generate_matched_qgaussian_bunch
from xnf4d.one_turn_map import OneTurnMap4D
from xnf4d.orbits import hamiltonian_orbits

LINE_FILE = "test_data/levelling.26_b1.json.gz"
ELEMENT = "tcp.c6l7.b1"
NEMITT_X = 3.5e-6
NEMITT_Y = 3.5e-6
SIGMA_Z = 7.61e-2
NUM_PARTICLES = 500_000
MAX_MAP_ORDER = 8
MAX_NF_ORDER = 7  # the normal form uses the map up to order MAX_NF_ORDER + 1
SIGMA_LIMIT = 8.0  # fit region of the one-turn map, beyond the cuts
Q, BETA, SIGMA_CUT = 1.5, 1.68, 5.0

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
# Bunches
# --------------------------------------------------------------------------- #

common = dict(line=line, normal_form=nf, at_element=ELEMENT, twiss=tw)
bunches = {
    "Gaussian": generate_matched_gaussian_bunch(
        NUM_PARTICLES, NEMITT_X, NEMITT_Y, SIGMA_Z, seed=1, **common
    )
}
for i, (fact, skew_cut) in enumerate([(True, True), (True, False), (False, True), (False, False)]):
    label = (
        f"q-Gaussian, {'factorisable' if fact else 'non-factorisable'}, "
        f"{'skew' if skew_cut else 'rectangular'} cut at {SIGMA_CUT:g}$\\sigma$"
    )
    bunches[label] = generate_matched_qgaussian_bunch(
        NUM_PARTICLES, NEMITT_X, NEMITT_Y, SIGMA_Z,
        q=Q, beta=BETA, fact=fact, sigma_cut_x=SIGMA_CUT, sigma_cut_y=SIGMA_CUT,
        skew_cut=skew_cut, seed=2 + i, **common,
    )

# --------------------------------------------------------------------------- #
# Orbits through (r, 0) in the normalised phase space of each plane, with the
# other plane at zero: Hamiltonian orbits and linear orbits (circles)
# --------------------------------------------------------------------------- #

radii = np.arange(1.0, SIGMA_CUT + 1.0)


def nonlinear_orbits(plane):
    """Hamiltonian orbits through (r, 0) in `plane`, for r in `radii`."""
    start = np.zeros((4, len(radii)))
    start[0 if plane == "x" else 2] = radii
    ho = hamiltonian_orbits(nf, start, grid_size=801, extent=1.3 * radii.max())
    return [c for curves in ho.curves_norm for c in curves]


theta_c = np.linspace(0, 2 * np.pi, 361)
orbits = {"x": nonlinear_orbits("x"), "y": nonlinear_orbits("y")}

# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #

lim = SIGMA_CUT + 1.0
edges = np.linspace(-lim, lim, 151)
panels = [  # (axis labels, coordinates, plane of the orbits)
    ((r"$\hat{x}$ [$\sigma$]", r"$\hat{p}_x$ [$\sigma$]"), ("x_norm", "px_norm"), "x"),
    ((r"$\hat{y}$ [$\sigma$]", r"$\hat{p}_y$ [$\sigma$]"), ("y_norm", "py_norm"), "y"),
    ((r"$\hat{x}$ [$\sigma$]", r"$\hat{y}$ [$\sigma$]"), ("x_norm", "y_norm"), None),
    ((r"$\hat{p}_x$ [$\sigma$]", r"$\hat{p}_y$ [$\sigma$]"), ("px_norm", "py_norm"), None),
]
normalised = {}

for label, particles in bunches.items():
    norm = tw.get_normalized_coordinates(particles, nemitt_x=NEMITT_X, nemitt_y=NEMITT_Y)
    normalised[label] = norm
    fig, axes = plt.subplots(1, 5, figsize=(30, 6), layout="constrained")
    for ax, (labels, (a, b), plane) in zip(axes, panels):
        *_, image = ax.hist2d(getattr(norm, a), getattr(norm, b), bins=[edges, edges],
                              cmap="Blues", norm=LogNorm())
        if plane is not None:
            i = 0 if plane == "x" else 2
            for rr in radii:
                ax.plot(rr * np.cos(theta_c), rr * np.sin(theta_c), color="tab:orange", ls="--", lw=1)
            for c in orbits[plane]:
                ax.plot(c[i], c[i + 1], color="tab:red", lw=1)
        ax.set_aspect("equal")
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        fig.colorbar(image, ax=ax, label="particles per bin", shrink=0.7)
    axes[0].plot([], [], color="tab:red", label="Hamiltonian orbits")
    axes[0].plot([], [], color="tab:orange", ls="--", label="linear orbits")
    axes[0].legend(loc="upper right", fontsize=8)

    ax = axes[4]
    *_, image = ax.hist2d(particles.zeta * 1e3, particles.delta * 1e4, bins=150,
                          cmap="Blues", norm=LogNorm())
    ax.set_xlabel(r"$\zeta$ [mm]")
    ax.set_ylabel(r"$\delta$ [$10^{-4}$]")
    fig.colorbar(image, ax=ax, label="particles per bin", shrink=0.7)
    fig.suptitle(label)

# distributions of x_norm of all bunches
fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
bins = np.linspace(-lim, lim, 121)
for label, norm in normalised.items():
    ax.hist(norm.x_norm, bins=bins, density=True, histtype="step", lw=1.5, label=label)
ax.set_yscale("log")
ax.set_xlabel(r"$\hat{x}$ [$\sigma$]")
ax.set_ylabel("probability density")
ax.set_title(f"Horizontal distributions (q = {Q:g}, beta = {BETA:g})")
ax.legend(fontsize=9)
plt.show()
