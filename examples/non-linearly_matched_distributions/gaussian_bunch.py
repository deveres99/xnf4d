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

    python examples/non-linearly_matched_distributions/gaussian_bunch.py
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import xpart as xp
import xtrack as xt

from xnf4d.distributions import generate_matched_gaussian_bunch
from xnf4d.one_turn_map import OneTurnMap4D
from xnf4d.orbits import hamiltonian_tori

LINE_FILE = "test_data/levelling.26_b1.json.gz"
ELEMENT = "tcp.c6l7.b1"
NEMITT_X = 3.5e-6
NEMITT_Y = 3.5e-6
SIGMA_Z = 7.61e-2
SIGMA_MAX = 5
NUM_PARTICLES = 200_000
MAX_MAP_ORDER = 8
MAX_NF_ORDER = 7  # the normal form uses the map up to order MAX_NF_ORDER + 1
SIGMA_LIMIT = 8.0  # fit region of the one-turn map, covering the Gaussian tails

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
# Generate Gaussian bunch
# --------------------------------------------------------------------------- #

particles = generate_matched_gaussian_bunch(
    NUM_PARTICLES, NEMITT_X, NEMITT_Y, SIGMA_Z,
    line=line, normal_form=nf, at_element=ELEMENT, twiss=tw, seed=2,
)
particles_norm = tw.get_normalized_coordinates(particles, nemitt_x=NEMITT_X, nemitt_y=NEMITT_Y)

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
# Plot initial distribution
# --------------------------------------------------------------------------- #
fs = 18

cmap = matplotlib.colormaps["viridis"]
cmap.set_under('white')

fig, axes = plt.subplots(2, 2, figsize=(10, 8), layout="tight")

*_, image00 = axes[0, 0].hist2d(
    particles.x * 1e3, particles.px * 1e6,
    bins=100,
    cmap=cmap, norm=LogNorm(vmin=1)
)
*_, image01 = axes[0, 1].hist2d(
    particles.y * 1e3, particles.py * 1e6,
    bins=100,
    cmap=cmap, norm=LogNorm(vmin=1)
)
*_, image10 = axes[1, 0].hist2d(
    particles_norm.x_norm, particles_norm.px_norm,
    bins=100,
    cmap=cmap, norm=LogNorm(vmin=1)
)
*_, image11 = axes[1, 1].hist2d(
    particles_norm.y_norm, particles_norm.py_norm,
    bins=100,
    cmap=cmap, norm=LogNorm(vmin=1)
)
for orb_phys, orb_norm, nl_orb_phys, nl_orb_norm in zip(orbits_phys_x, orbits_norm_x, nl_orbits_phys_x, nl_orbits_norm_x):
    axes[0, 0].plot(
        nl_orb_phys[0] * 1e3, nl_orb_phys[1] * 1e6,
        color="red", linestyle='-', linewidth=2
    )
    axes[0, 0].plot(
        orb_phys[0] * 1e3, orb_phys[1] * 1e6,
        color="orange", linestyle='--', linewidth=2
    )
    axes[1, 0].plot(
        nl_orb_norm[0], nl_orb_norm[1],
        color="red", linestyle='-', linewidth=2
    )
    axes[1, 0].plot(
        orb_norm[0], orb_norm[1],
        color="orange", linestyle='--', linewidth=2
    )
for orb_phys, orb_norm, nl_orb_phys, nl_orb_norm in zip(orbits_phys_y, orbits_norm_y, nl_orbits_phys_y, nl_orbits_norm_y):
    axes[0, 1].plot(
        nl_orb_phys[2] * 1e3, nl_orb_phys[3] * 1e6,
        color="red", linestyle='-', linewidth=2
    )
    axes[0, 1].plot(
        orb_phys[2] * 1e3, orb_phys[3] * 1e6,
        color="orange", linestyle='--', linewidth=2
    )
    axes[1, 1].plot(
        nl_orb_norm[2], nl_orb_norm[3],
        color="red", linestyle='-', linewidth=2
    )
    axes[1, 1].plot(
        orb_norm[2], orb_norm[3],
        color="orange", linestyle='--', linewidth=2
    )

axes[0, 0].set_xlim(-sigma_x * 1e3 * (SIGMA_MAX+1), sigma_x * 1e3 * (SIGMA_MAX+1))
axes[0, 0].set_ylim(-sigma_px * 1e6 * (SIGMA_MAX+1), sigma_px * 1e6 * (SIGMA_MAX+1))
axes[0, 0].set_xlabel(r"$x$ [mm]", fontsize=fs)
axes[0, 0].set_ylabel(r"$p_x$ [$\mu$rad]", fontsize=fs)
axes[0, 0].tick_params(axis="both", labelsize=fs-2)
cbar00 = fig.colorbar(image00, ax=axes[0, 0])
cbar00.ax.tick_params(labelsize=fs-2)
cbar00.set_label("particles per bin", fontsize=fs)

axes[0, 1].set_xlim(-sigma_y * 1e3 * (SIGMA_MAX+1), sigma_y * 1e3 * (SIGMA_MAX+1))
axes[0, 1].set_ylim(-sigma_py * 1e6 * (SIGMA_MAX+1), sigma_py * 1e6 * (SIGMA_MAX+1))
axes[0, 1].set_xlabel(r"$y$ [mm]", fontsize=fs)
axes[0, 1].set_ylabel(r"$p_y$ [$\mu$rad]", fontsize=fs)
axes[0, 1].tick_params(axis="both", labelsize=fs-2)
cbar01 = fig.colorbar(image01, ax=axes[0, 1])
cbar01.ax.tick_params(labelsize=fs-2)
cbar01.set_label("particles per bin", fontsize=fs)

axes[1, 0].set_xlim(-(SIGMA_MAX+1), (SIGMA_MAX+1))
axes[1, 0].set_ylim(-(SIGMA_MAX+1), (SIGMA_MAX+1))
axes[1, 0].set_xlabel(r"$\hat{x}$ [$\sigma$]", fontsize=fs)
axes[1, 0].set_ylabel(r"$\hat{p}_x$ [$\sigma$]", fontsize=fs)
axes[1, 0].tick_params(axis="both", labelsize=fs-2)
cbar10 = fig.colorbar(image10, ax=axes[1, 0])
cbar10.ax.tick_params(labelsize=fs-2)
cbar10.set_label("particles per bin", fontsize=fs)

axes[1, 1].set_xlim(-(SIGMA_MAX+1), (SIGMA_MAX+1))
axes[1, 1].set_ylim(-(SIGMA_MAX+1), (SIGMA_MAX+1))
axes[1, 1].set_xlabel(r"$\hat{y}$ [$\sigma$]", fontsize=fs)
axes[1, 1].set_ylabel(r"$\hat{p}_y$ [$\sigma$]", fontsize=fs)
axes[1, 1].tick_params(axis="both", labelsize=fs-2)
cbar11 = fig.colorbar(image11, ax=axes[1, 1])
cbar11.ax.tick_params(labelsize=fs-2)
cbar11.set_label("particles per bin", fontsize=fs)

axes[0, 0].plot([], [], color="red", linestyle='-', linewidth=2, label='non-linear invariant tori')
axes[0, 0].plot([], [], color="orange", linestyle='--', linewidth=2, label='linear invariant tori')
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.0),
    ncol=len(labels),
    fontsize=fs-2,
    columnspacing=0.8,  # Space between columns
    handletextpad=0.3,  # Space between handle and label
)
fig.tight_layout(rect=[0, 0, 1, 0.94])

fig, ax = plt.subplots(1, 1, figsize=(6, 4), layout="tight")

*_, image = ax.hist2d(
    particles.zeta * 1e3, particles.delta * 1e3,
    bins=100,
    cmap=cmap, norm=LogNorm(vmin=1)
)

ax.set_xlabel(r"$\zeta$ [mm]", fontsize=fs)
ax.set_ylabel(r"$\delta$ [$10^{-3}$]", fontsize=fs)
ax.tick_params(axis="both", labelsize=fs-2)
cbar = fig.colorbar(image, ax=ax)
cbar.ax.tick_params(labelsize=fs-2)
cbar.set_label("particles per bin", fontsize=fs)

plt.show()