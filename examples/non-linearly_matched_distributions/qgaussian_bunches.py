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

    python examples/non-linearly_matched_distributions/qgaussian_bunches.py
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import xpart as xp
import xtrack as xt

from xnf4d.distributions import generate_matched_gaussian_bunch, generate_matched_qgaussian_bunch
from xnf4d.one_turn_map import OneTurnMap4D
from xnf4d.orbits import hamiltonian_tori

LINE_FILE = "test_data/levelling.26_b1.json.gz"
ELEMENT = "tcp.c6l7.b1"
NEMITT_X = 3.5e-6
NEMITT_Y = 3.5e-6
SIGMA_Z = 7.61e-2
SIGMA_MAX = 5
NUM_PARTICLES = 500_000
MAX_MAP_ORDER = 8
MAX_NF_ORDER = 7  # the normal form uses the map up to order MAX_NF_ORDER + 1
SIGMA_LIMIT = 8.0  # fit region of the one-turn map, beyond the cuts
Q, BETA, SIGMA_CUT = 1.5, 1.68, 5.0

# --------------------------------------------------------------------------- #
# Load and set up line
# --------------------------------------------------------------------------- #

line = xt.load(LINE_FILE)
line.cycle(name_first_element=ELEMENT)
line["dqx.b1_op"] = 0.0
line["dqy.b1_op"] = 0.0
line.build_tracker()
tw = line.twiss()
beam_sizes = tw.get_beam_covariance(
    nemitt_x=NEMITT_X, nemitt_y=NEMITT_Y,
    gemitt_zeta=SIGMA_Z**2 / tw.bets0
)
sigma_x = beam_sizes.sigma_x[0]
sigma_px = beam_sizes.sigma_px[0]
sigma_y = beam_sizes.sigma_y[0]
sigma_py = beam_sizes.sigma_py[0]

# --------------------------------------------------------------------------- #
# Compute one-turn map and normal form at the start of line (i.e. at collimator)
# --------------------------------------------------------------------------- #

otm = OneTurnMap4D(line, MAX_MAP_ORDER, NEMITT_X, NEMITT_Y, sigma_limit=SIGMA_LIMIT, seed=1)
otm.compute()
nf = otm.calculate_normal_form(MAX_NF_ORDER)  # non-resonant

# --------------------------------------------------------------------------- #
# Get Hamiltonian orbits through points on the x_norm axis (y = py = 0), in 
# normalised coordinates (x_norm, px_norm, y_norm, py_norm) and transform into 
# physical coordinates
# --------------------------------------------------------------------------- #

gamma_beta = tw.particle_on_co.gamma0[0] * tw.particle_on_co.beta0[0]
W = tw.W_matrix[0][:4, :4] * np.sqrt(np.array([NEMITT_X, NEMITT_X, NEMITT_Y, NEMITT_Y]) / gamma_beta)
co = np.array([tw.x[0], tw.px[0], tw.y[0], tw.py[0]])

radii = np.arange(1.0, SIGMA_MAX+1.5, 1.0)

initial_coords_x = np.zeros((4, len(radii)))
initial_coords_x[0] = radii
ht_x = hamiltonian_tori(nf, initial_coords_x, num_points=4096)
nl_orbits_norm_x = [
    np.concatenate([tt := np.array([row.ravel() for row in t]), tt[:, :1]], axis=1)
    for tori in ht_x.tori_norm
    for t in tori
]
nl_orbits_phys_x = [
    np.expand_dims(co, axis=1) + np.matmul(W, orb) for orb in nl_orbits_norm_x
]

# --------------------------------------------------------------------------- #
# Get Hamiltonian orbits through points on the y_norm axis (x = px = 0), in 
# normalised coordinates (x_norm, px_norm, y_norm, py_norm) and transform into 
# physical coordinates
# --------------------------------------------------------------------------- #

initial_coords_y = np.zeros((4, len(radii)))
initial_coords_y[2] = radii
ht_y = hamiltonian_tori(nf, initial_coords_y, num_points=4096)
nl_orbits_norm_y = [
    np.concatenate([tt := np.array([row.ravel() for row in t]), tt[:, :1]], axis=1)
    for tori in ht_y.tori_norm
    for t in tori
]
nl_orbits_phys_y = [
    np.expand_dims(co, axis=1) + np.matmul(W, orb) for orb in nl_orbits_norm_y
]

# --------------------------------------------------------------------------- #
# Get linear orbits through points on the x_norm axis (y = py = 0), in 
# normalised coordinates (x_norm, px_norm, y_norm, py_norm) and transform into 
# physical coordinates
# --------------------------------------------------------------------------- #

theta = np.linspace(0, 2*np.pi, 1000)
orbits_norm_x = [
    np.vstack((r * np.cos(theta), r * np.sin(theta), np.zeros(len(theta)), np.zeros(len(theta))))
    for r in radii
]
orbits_phys_x = [
    np.expand_dims(co, axis=1) + np.matmul(W, orb) for orb in orbits_norm_x
]

# --------------------------------------------------------------------------- #
# Get linear orbits through points on the y_norm axis (x = px = 0), in 
# normalised coordinates (x_norm, px_norm, y_norm, py_norm) and transform into 
# physical coordinates
# --------------------------------------------------------------------------- #

orbits_norm_y = [
    np.vstack((np.zeros(len(theta)), np.zeros(len(theta)), r * np.cos(theta), r * np.sin(theta)))
    for r in radii
]
orbits_phys_y = [
    np.expand_dims(co, axis=1) + np.matmul(W, orb) for orb in orbits_norm_y
]

# --------------------------------------------------------------------------- #
# Generate bunches
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# Bunches
# --------------------------------------------------------------------------- #

common = dict(line=line, normal_form=nf, at_element=ELEMENT, twiss=tw)
bunches = {
    "Gaussian": generate_matched_gaussian_bunch(
        NUM_PARTICLES, NEMITT_X, NEMITT_Y, SIGMA_Z, seed=1, **common
    )
}
for i, (fact, skew_cut) in enumerate([(True, False), (True, True), (False, False), (False, True)]):
    label = (
        f"q-Gaussian, {'factorisable' if fact else 'non-factorisable'}, "
        f"cut at {SIGMA_CUT:g}$\\sigma$ {'w/ skew' if skew_cut else 'w/o skew'} "
    )
    bunches[label] = generate_matched_qgaussian_bunch(
        NUM_PARTICLES, NEMITT_X, NEMITT_Y, SIGMA_Z,
        q=Q, beta=BETA, fact=fact, sigma_cut_x=SIGMA_CUT, sigma_cut_y=SIGMA_CUT,
        skew_cut=skew_cut, seed=2+i, **common,
    )
bunches_norm = {}
for key, val in bunches.items():
    bunches_norm[key] = tw.get_normalized_coordinates(val, nemitt_x=NEMITT_X, nemitt_y=NEMITT_Y)

# --------------------------------------------------------------------------- #
# Plot initial distributions
# --------------------------------------------------------------------------- #
fs = 18

cmap = matplotlib.colormaps["viridis"]
cmap.set_under('white')

colors = ["black", "blue", "cornflowerblue", "red", "coral"]

def plot(particles, particles_norm, title):
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), layout="tight")

    *_, image0 = axes[0].hist2d(
        particles_norm.x_norm, particles_norm.px_norm,
        bins=100,
        cmap=cmap, norm=LogNorm(vmin=1)
    )
    *_, image1 = axes[1].hist2d(
        particles_norm.y_norm, particles_norm.py_norm,
        bins=100,
        cmap=cmap, norm=LogNorm(vmin=1)
    )
    *_, image2 = axes[2].hist2d(
        particles.zeta * 1e3, particles.delta * 1e3,
        bins=100,
        cmap=cmap, norm=LogNorm(vmin=1)
    )
    *_, image3 = axes[3].hist2d(
        particles_norm.x_norm, particles_norm.y_norm,
        bins=100,
        cmap=cmap, norm=LogNorm(vmin=1)
    )
    for orb_norm,nl_orb_norm in zip(orbits_norm_x, nl_orbits_norm_x):
        axes[0].plot(
            nl_orb_norm[0], nl_orb_norm[1],
            color="red", linestyle='-', linewidth=2
        )
        axes[0].plot(
            orb_norm[0], orb_norm[1],
            color="orange", linestyle='--', linewidth=2
        )
    for orb_norm, nl_orb_norm in zip(orbits_norm_y, nl_orbits_norm_y):
        axes[1].plot(
            nl_orb_norm[2], nl_orb_norm[3],
            color="red", linestyle='-', linewidth=2
        )
        axes[1].plot(
            orb_norm[2], orb_norm[3],
            color="orange", linestyle='--', linewidth=2
        )

    axes[0].set_xlim((SIGMA_MAX+1), -(SIGMA_MAX+1))
    axes[0].set_ylim((SIGMA_MAX+1), -(SIGMA_MAX+1))
    axes[0].set_xlabel(r"$\hat{x}$ [$\sigma$]", fontsize=fs)
    axes[0].set_ylabel(r"$\hat{p}_x$ [$\sigma$]", fontsize=fs)
    axes[0].tick_params(axis="both", labelsize=fs-2)
    cbar0 = fig.colorbar(image0, ax=axes[0])
    cbar0.ax.tick_params(labelsize=fs-2)
    cbar0.set_label("particles per bin", fontsize=fs)

    axes[1].set_xlim((SIGMA_MAX+1), -(SIGMA_MAX+1))
    axes[1].set_ylim((SIGMA_MAX+1), -(SIGMA_MAX+1))
    axes[1].set_xlabel(r"$\hat{y}$ [$\sigma$]", fontsize=fs)
    axes[1].set_ylabel(r"$\hat{p}_y$ [$\sigma$]", fontsize=fs)
    axes[1].tick_params(axis="both", labelsize=fs-2)
    cbar1 = fig.colorbar(image1, ax=axes[1])
    cbar1.ax.tick_params(labelsize=fs-2)
    cbar1.set_label("particles per bin", fontsize=fs)

    axes[2].set_xlabel(r"$\zeta$ [mm]", fontsize=fs)
    axes[2].set_ylabel(r"$\delta$ [$10^{-3}$]", fontsize=fs)
    axes[2].tick_params(axis="both", labelsize=fs-2)
    cbar2 = fig.colorbar(image2, ax=axes[2])
    cbar2.ax.tick_params(labelsize=fs-2)
    cbar2.set_label("particles per bin", fontsize=fs)

    axes[3].set_xlim((SIGMA_MAX+1), -(SIGMA_MAX+1))
    axes[3].set_ylim((SIGMA_MAX+1), -(SIGMA_MAX+1))
    axes[3].set_xlabel(r"$\hat{x}$ [$\sigma$]", fontsize=fs)
    axes[3].set_ylabel(r"$\hat{y}$ [$\sigma$]", fontsize=fs)
    axes[3].tick_params(axis="both", labelsize=fs-2)
    cbar3 = fig.colorbar(image3, ax=axes[3])
    cbar3.ax.tick_params(labelsize=fs-2)
    cbar3.set_label("particles per bin", fontsize=fs)

    axes[0].plot([], [], color="red", linestyle='-', linewidth=2, label='non-linear invariant tori')
    axes[0].plot([], [], color="orange", linestyle='--', linewidth=2, label='linear invariant tori')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=len(labels),
        fontsize=fs-2,
        columnspacing=0.8,  # Space between columns
        handletextpad=0.3,  # Space between handle and label
    )
    fig.tight_layout(rect=[0, 0.1, 1, 0.94])

    fig.suptitle(title, fontsize=fs)

for key in bunches.keys():
    plot(bunches[key], bunches_norm[key], key)

fig, ax = plt.subplots(1, 1, figsize=(12, 6), layout="tight")

bins = 100
hist_range = [-SIGMA_MAX, SIGMA_MAX]

for i, (key, b_norm) in enumerate(bunches_norm.items()):
    ax.hist(
        b_norm.x_norm,
        bins=bins, range=hist_range, histtype="step", density=True,
        color=colors[i], label=key
    )

ax.set_ylim(1e-5, 1)
ax.set_xlabel(r"$\hat{x}$ [$\sigma$]", fontsize=fs)
ax.set_ylabel("particle density", fontsize=fs)
ax.tick_params(axis="both", labelsize=fs-2)
ax.set_yscale("log")
ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.01),
    ncol=2,
    fontsize=fs-2,
    columnspacing=0.8,  # Space between columns
    handletextpad=0.3,  # Space between handle and label
)
fig.tight_layout(rect=[0, 0, 1, 0.98])

plt.show()