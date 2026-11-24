"""
Non-linearly matched pencil beam on the LHC horizontal primary collimator.

The pencil is generated at ``tcp.c6l7.b1`` with jaws at +/- 5 sigma from the
closed orbit, using a non-resonant normal form of the one-turn map at the
collimator. It is plotted in the horizontal phase space, in physical and in
Courant-Snyder normalised coordinates, together with invarian tori of the normal 
form Hamiltonian, which the pencil follows, and the invariant tori of the linear 
optics, i.e. ellipses. With linear coupling the jaw is not a single line in 
normalised coordinates: it is drawn as a band of lines, one per vertical 
amplitude and phase.

Run from the repository root, with ``xnf4d`` installed (``pip install
".[examples]"``, see the README)::

    python examples/non-linearly_matched_distributions/pencil_on_collimator.py
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import xtrack as xt

from xnf4d.distributions import generate_pencil_on_collimator
from xnf4d.one_turn_map import OneTurnMap4D
from xnf4d.orbits import hamiltonian_tori

LINE_FILE = "test_data/levelling.26_b1.json.gz"
COLLIMATOR = "tcp.c6l7.b1"
NEMITT_X = 3.5e-6
NEMITT_Y = 3.5e-6
SIGMA_Z = 7.61e-2
JAW_SIGMAS = 5.0
SPREAD_SIGMAS = 0.1  # pencil spread, made large for visibility (xcoll default: 1 um)
NUM_PARTICLES = 10000
MAX_MAP_ORDER = 8
MAX_NF_ORDER = 7  # the normal form uses the map up to order MAX_NF_ORDER + 1
SIGMA_LIMIT = 8.0  # fit region of the one-turn map, well beyond the jaw

# --------------------------------------------------------------------------- #
# Load and set up line
# --------------------------------------------------------------------------- #

line = xt.load(LINE_FILE)
line.cycle(name_first_element=COLLIMATOR)
line["dqx.b1_op"] = 0.0
line["dqy.b1_op"] = 0.0
line.build_tracker()
tw = line.twiss()

# --------------------------------------------------------------------------- #
# Compute one-turn map and normal form at the start of line (i.e. at collimator)
# --------------------------------------------------------------------------- #

otm = OneTurnMap4D(line, MAX_MAP_ORDER, NEMITT_X, NEMITT_Y, sigma_limit=SIGMA_LIMIT, seed=1)
otm.compute()
nf = otm.calculate_normal_form(MAX_NF_ORDER)  # non-resonant

# --------------------------------------------------------------------------- #
# Generate pencil on both jaws
# --------------------------------------------------------------------------- #

beam_sizes = tw.get_beam_covariance(
    nemitt_x=NEMITT_X, nemitt_y=NEMITT_Y,
    gemitt_zeta=SIGMA_Z**2 / tw.bets0
)
sigma_x = beam_sizes.sigma_x[0]
sigma_px = beam_sizes.sigma_px[0]
sigma_y = beam_sizes.sigma_y[0]
sigma_py = beam_sizes.sigma_py[0]
jaw_L = tw.x[0] + JAW_SIGMAS * sigma_x
jaw_R = tw.x[0] - JAW_SIGMAS * sigma_x

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
particles_norm = tw.get_normalized_coordinates(particles, nemitt_x=NEMITT_X, nemitt_y=NEMITT_Y)

# --------------------------------------------------------------------------- #
# Get Hamiltonian orbits through points on the x_norm axis (y = py = 0), in 
# normalised coordinates (x_norm, px_norm, y_norm, py_norm) and transform into 
# physical coordinates
# --------------------------------------------------------------------------- #

gamma_beta = tw.particle_on_co.gamma0[0] * tw.particle_on_co.beta0[0]
W = tw.W_matrix[0][:4, :4] * np.sqrt(np.array([NEMITT_X, NEMITT_X, NEMITT_Y, NEMITT_Y]) / gamma_beta)
co = np.array([tw.x[0], tw.px[0], tw.y[0], tw.py[0]])

radii = np.hstack((
    np.arange(1.0, JAW_SIGMAS + 2.0),
    [JAW_SIGMAS-2*SPREAD_SIGMAS, JAW_SIGMAS+2*SPREAD_SIGMAS],
    [-JAW_SIGMAS-2*SPREAD_SIGMAS, -JAW_SIGMAS+2*SPREAD_SIGMAS],
))
initial_coords = np.zeros((4, len(radii)))
initial_coords[0] = radii
ht = hamiltonian_tori(nf, initial_coords, num_points=4096)
nl_orbits_norm = [
    np.concatenate([tt := np.array([row.ravel() for row in t]), tt[:, :1]], axis=1)
    for tori in ht.tori_norm
    for t in tori
]
nl_orbits_phys = [
    np.expand_dims(co, axis=1) + np.matmul(W, orb) for orb in nl_orbits_norm
]


# --------------------------------------------------------------------------- #
# Get linear orbits through points on the x_norm axis (y = py = 0), in 
# normalised coordinates (x_norm, px_norm, y_norm, py_norm) and transform into 
# physical coordinates
# --------------------------------------------------------------------------- #

theta = np.linspace(0, 2*np.pi, 1000)
orbits_norm = [
    np.vstack((r * np.cos(theta), r * np.sin(theta), np.zeros(len(theta)), np.zeros(len(theta))))
    for r in radii
]
orbits_phys = [
    np.expand_dims(co, axis=1) + np.matmul(W, orb) for orb in orbits_norm
]


# --------------------------------------------------------------------------- #
# There is some coupling, so jaw is a single straight line only in physical 
# coordinates, in normalised coordinates there are many lines corresponding to 
# different y, py physical coordinates, forming a band. Compute jaw band in 
# normalised coordinates.
# --------------------------------------------------------------------------- #

jaw_pts = 101  # points per line (along px)
jaw_phi = np.linspace(0, 2*np.pi, 64, endpoint=False)  # vertical phases, one line each
# physical (y, py) on the linear vertical ellipse of JAW_SIGMAS: its lines bound the band
jaw_ypy = co[2:, None] + W[2:, 2:] @ (JAW_SIGMAS * np.vstack((np.cos(jaw_phi), np.sin(jaw_phi))))
px_line = np.linspace(-(2 * JAW_SIGMAS) * sigma_px, (2 * JAW_SIGMAS) * sigma_px, jaw_pts)
# px swept back and forth, so that all lines form a single polyline for plt.plot
jaw_px = tw.px[0] + np.concatenate([px_line[::(-1)**i] for i in range(len(jaw_phi))])
jaw_y, jaw_py = np.repeat(jaw_ypy[0], jaw_pts), np.repeat(jaw_ypy[1], jaw_pts)
jaw_L_x = np.full(jaw_px.size, fill_value=jaw_L)
jaw_R_x = np.full(jaw_px.size, fill_value=jaw_R)

jaw_band_L_phys = line.build_particles(
    zeta=tw.particle_on_co.zeta[0],
    delta=tw.particle_on_co.delta[0],
    x=jaw_L_x, px=jaw_px, y=jaw_y, py=jaw_py
)
jaw_band_L_norm = tw.get_normalized_coordinates(jaw_band_L_phys, nemitt_x=NEMITT_X, nemitt_y=NEMITT_Y)
jaw_band_R_phys = line.build_particles(
    zeta=tw.particle_on_co.zeta[0],
    delta=tw.particle_on_co.delta[0],
    x=jaw_R_x, px=jaw_px, y=jaw_y, py=jaw_py
)
jaw_band_R_norm = tw.get_normalized_coordinates(jaw_band_R_phys, nemitt_x=NEMITT_X, nemitt_y=NEMITT_Y)

# --------------------------------------------------------------------------- #
# Plot initial distribution
# --------------------------------------------------------------------------- #
fs = 18

fig, axes = plt.subplots(2, 3, figsize=(12, 8), layout="tight")

for i in range(3):
    if i == 0:
        axes[0, i].axvline(jaw_L * 1e3, color="black", linewidth=3, label="jaw edge")
        axes[1, i].axvspan(
            np.min(jaw_band_L_norm.x_norm), np.max(jaw_band_L_norm.x_norm),
            color="black", linewidth=3, alpha=0.5, label="jaw edge"
        )
    else:
        axes[0, i].axvline(jaw_L * 1e3, color="black", linewidth=3)
        axes[1, i].axvspan(
            np.min(jaw_band_L_norm.x_norm), np.max(jaw_band_L_norm.x_norm),
            color="black", linewidth=3, alpha=0.5
        )
    axes[0, i].axvline(jaw_R * 1e3, color="black", linewidth=3)
    axes[1, i].axvspan(
        np.min(jaw_band_R_norm.x_norm), np.max(jaw_band_R_norm.x_norm),
        color="black", linewidth=3, alpha=0.5
    )
    axes[0, i].scatter(
        particles.x * 1e3, particles.px * 1e6,
        c="blue", marker="o"
    )
    axes[0, i].set_xlabel(r"$x$ [mm]", fontsize=fs)
    axes[0, i].set_ylabel(r"$p_x$ [$\mu$rad]", fontsize=fs)
    axes[0, i].tick_params(axis="both", labelsize=fs-2)
    axes[1, i].scatter(
        particles_norm.x_norm, particles_norm.px_norm,
        c="blue", marker="o"
    )
    axes[1, i].set_xlabel(r"$\hat{x}$ [$\sigma$]", fontsize=fs)
    axes[1, i].set_ylabel(r"$\hat{p}_x$ [$\sigma$]", fontsize=fs)
    axes[1, i].tick_params(axis="both", labelsize=fs-2)
n = 0
for orb_phys, orb_norm, nl_orb_phys, nl_orb_norm in zip(orbits_phys, orbits_norm, nl_orbits_phys, nl_orbits_norm):
    if n < len(orbits_phys) - 4:
        rng = range(3)
    else:
        rng = range(1, 3)
    for i in rng:
        axes[0, i].plot(
            nl_orb_phys[0] * 1e3, nl_orb_phys[1] * 1e6,
            color="red", linestyle='-', linewidth=2
        )
        axes[0, i].plot(
            orb_phys[0] * 1e3, orb_phys[1] * 1e6,
            color="orange", linestyle='--', linewidth=2
        )
        axes[1, i].plot(
            nl_orb_norm[0], nl_orb_norm[1],
            color="red", linestyle='-', linewidth=2
        )
        axes[1, i].plot(
            orb_norm[0], orb_norm[1],
            color="orange", linestyle='--', linewidth=2
        )
    n += 1
mask_L = particles.x > 0
mask_R = particles.x < 0
axes[0, 1].set_xlim((jaw_L-0.5*sigma_x)*1e3, (jaw_L+sigma_x)*1e3)
axes[0, 1].set_ylim((
    (np.mean(particles.px[mask_L])-5*np.std(particles.px[mask_L]))*1e6,
    (np.mean(particles.px[mask_L])+5*np.std(particles.px[mask_L]))*1e6
))
axes[0, 2].set_xlim((jaw_R-0.5*sigma_x)*1e3, (jaw_R+sigma_x)*1e3)
axes[0, 2].set_ylim((
    (np.mean(particles.px[mask_R])-5*np.std(particles.px[mask_R]))*1e6,
    (np.mean(particles.px[mask_R])+5*np.std(particles.px[mask_R]))*1e6
))
axes[1, 0].set_xlim(-JAW_SIGMAS-1.5, JAW_SIGMAS+1.5)
axes[1, 0].set_ylim(-JAW_SIGMAS-1.5, JAW_SIGMAS+1.5)
axes[1, 1].set_xlim(JAW_SIGMAS-0.5, JAW_SIGMAS+0.5)
axes[1, 1].set_ylim((
    np.mean(particles_norm.px_norm[mask_L])-5*np.std(particles_norm.px_norm[mask_L]),
    np.mean(particles_norm.px_norm[mask_L])+5*np.std(particles_norm.px_norm[mask_L])
))
axes[1, 2].set_xlim(-JAW_SIGMAS-0.5, -JAW_SIGMAS+0.5)
axes[1, 2].set_ylim((
    np.mean(particles_norm.px_norm[mask_R])-5*np.std(particles_norm.px_norm[mask_R]),
    np.mean(particles_norm.px_norm[mask_R])+5*np.std(particles_norm.px_norm[mask_R])
))

axes[0, 0].legend(loc="upper center", fontsize=fs-2)
axes[1, 0].legend(loc="upper center", fontsize=fs-2)

axes[0, 1].plot([], [], color="red", linestyle='-', linewidth=2, label='non-linear invariant tori')
axes[0, 1].plot([], [], color="orange", linestyle='--', linewidth=2, label='linear invariant tori')
handles, labels = axes[0, 1].get_legend_handles_labels()
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


cmap = matplotlib.colormaps["viridis"]
cmap.set_under('white')

fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="tight")

axes[0].hist2d(
    particles.y * 1e3, particles.py * 1e6,
    bins=100,
    cmap=cmap, vmin=1
)
axes[0].set_xlabel(r"$y$ [mm]", fontsize=fs)
axes[0].set_ylabel(r"$p_y$ [$\mu$rad]", fontsize=fs)
axes[0].tick_params(axis="both", labelsize=fs-2)

axes[1].hist2d(
    particles_norm.y_norm, particles_norm.py_norm,
    bins=100,
    cmap=cmap, vmin=1
)
axes[1].set_xlabel(r"$\hat{y}$ [$\sigma$]", fontsize=fs)
axes[1].set_ylabel(r"$\hat{p}_y$ [$\sigma$]", fontsize=fs)
axes[1].tick_params(axis="both", labelsize=fs-2)

axes[2].scatter(
    particles.zeta * 1e3, particles.delta * 1e3,
    c="blue", marker="o"
)
axes[2].set_xlabel(r"$\zeta$ [mm]", fontsize=fs)
axes[2].set_ylabel(r"$\delta$ [$10^{-3}$]", fontsize=fs)
axes[2].tick_params(axis="both", labelsize=fs-2)

plt.show()