"""
Orbits of an ``xtrack`` line obtained by tracking.

Initial conditions are placed in the normalised phase space of the
horizontal and/or vertical plane, along lines at fixed angles (by default
0, 18, 36, 54, 72 and 90 degrees from the ``x_norm`` (or ``y_norm``) axis),
at radii ``radial_step, 2 radial_step, ...`` up to ``sigma_limit``. The
normalised coordinates of the other plane are zero and the longitudinal
coordinates are those of the closed orbit. The particles are tracked and
their turn-by-turn coordinates returned as an :class:`Orbits` object, in
physical, Courant-Snyder normalised and optionally normal form coordinates.

Normalised coordinates follow ``xtrack``: deviations from the closed orbit,
transformed with the inverse of the twiss ``W`` matrix and divided by the
beam size ``sqrt(nemitt / (beta0 gamma0))``. These are the coordinates used
by :class:`~xnf4d.one_turn_map.OneTurnMap4D`, so a normal form computed from
such a map (with the same emittances and twiss method) can be used for the
normal form coordinates.

:func:`hamiltonian_orbits` gives the orbits predicted by the interpolating
Hamiltonian of a normal form, as its contours through the same initial
conditions, for comparison with tracking (see :class:`HamiltonianOrbits`).
:func:`hamiltonian_tori` gives the full 4D invariant tori instead, which also
describe the exchange of energy between coupled planes (see
:class:`HamiltonianTori`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, fields
from typing import Any, Literal

import warnings

import contourpy
import numpy as np

from .numerical_normal_form import NormalForm4D, complex_to_real, real_to_complex

__all__ = [
    "DEFAULT_ANGLES",
    "HamiltonianOrbits",
    "HamiltonianTori",
    "Orbits",
    "hamiltonian_orbits",
    "hamiltonian_tori",
    "initial_conditions",
    "track_orbits",
]

#: Default angles [deg] of the lines of initial conditions in phase space.
DEFAULT_ANGLES: tuple[float, ...] = (0.0, 18.0, 36.0, 54.0, 72.0, 90.0)

Plane = Literal["x", "y", "both"]

_PHYS = ("x", "px", "y", "py")
_NORM = ("x_norm", "px_norm", "y_norm", "py_norm")
_NF = ("x_nf", "px_nf", "y_nf", "py_nf")


@dataclass
class Orbits:
    """
    Turn-by-turn coordinates of tracked particles.

    Not meant to be created directly: it is returned by
    :func:`track_orbits`, e.g.
    ``orb = track_orbits(line, nemitt_x, nemitt_y, plane="x")``.

    Every coordinate is an array of shape ``(num_particles, num_turns)``:
    row ``i`` is the orbit of particle ``i`` and column ``k`` holds its
    coordinates after ``k`` turns (column ``0`` is the initial condition).
    For a particle lost during turn ``k``, columns ``k + 1`` onwards are
    ``nan``.

    Attributes
    ----------
    plane : numpy.ndarray of str, shape (num_particles,)
        Plane (``"x"`` or ``"y"``) in which each particle was started.
    radius : numpy.ndarray, shape (num_particles,)
        Initial distance from the closed orbit in normalised phase space
        [beam sizes].
    angle : numpy.ndarray, shape (num_particles,)
        Initial angle in normalised phase space [deg], measured from the
        ``x_norm`` (or ``y_norm``) axis towards ``px_norm`` (or ``py_norm``).
    lost : numpy.ndarray of bool, shape (num_particles,)
        Whether each particle was lost.
    num_valid_turns : numpy.ndarray of int, shape (num_particles,)
        Number of valid (non-``nan``) columns of each orbit.
    x, px, y, py : numpy.ndarray
        Physical coordinates.
    x_norm, px_norm, y_norm, py_norm : numpy.ndarray
        Courant-Snyder normalised coordinates.
    x_nf, px_nf, y_nf, py_nf : numpy.ndarray or None
        Real normal form coordinates (``x_nf - i px_nf`` and
        ``x_nf + i px_nf`` are the complex normal form coordinates
        ``zeta1, zeta1*``, and similarly for ``y``), if requested.
    """

    plane: np.ndarray
    radius: np.ndarray
    angle: np.ndarray
    lost: np.ndarray
    num_valid_turns: np.ndarray
    x: np.ndarray
    px: np.ndarray
    y: np.ndarray
    py: np.ndarray
    x_norm: np.ndarray
    px_norm: np.ndarray
    y_norm: np.ndarray
    py_norm: np.ndarray
    x_nf: np.ndarray | None = None
    px_nf: np.ndarray | None = None
    y_nf: np.ndarray | None = None
    py_nf: np.ndarray | None = None

    @property
    def num_particles(self) -> int:
        """
        Number of particles.

        Returns
        -------
        int
            Number of rows of the coordinate arrays.
        """
        return int(self.x.shape[0])

    @property
    def num_turns(self) -> int:
        """
        Number of turns recorded.

        Returns
        -------
        int
            Number of columns of the coordinate arrays.
        """
        return int(self.x.shape[1])

    @property
    def has_normal_form(self) -> bool:
        """
        Whether normal form coordinates are available.

        Returns
        -------
        bool
            ``True`` if ``x_nf`` etc. are set.
        """
        return self.x_nf is not None

    def select(self, plane: Literal["x", "y"]) -> Orbits:
        """
        Orbits of the particles started in one plane.

        Parameters
        ----------
        plane : {"x", "y"}
            Plane to select.

        Returns
        -------
        Orbits
            The orbits of these particles only.

        Raises
        ------
        ValueError
            If ``plane`` is not ``"x"`` or ``"y"``.
        """
        if plane not in ("x", "y"):
            raise ValueError(f"plane must be 'x' or 'y', got {plane!r}.")
        mask = self.plane == plane
        values = {}
        for f in fields(self):
            v = getattr(self, f.name)
            values[f.name] = None if v is None else v[mask]
        return Orbits(**values)


def initial_conditions(
    plane: Plane = "both",
    sigma_limit: float = 10.0,
    radial_step: float = 1.0,
    angles: Sequence[float] = DEFAULT_ANGLES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Initial conditions along lines in normalised phase space.

    For each plane and each angle, the radii are ``radial_step``,
    ``2 radial_step``, ... up to ``sigma_limit`` (included).

    Parameters
    ----------
    plane : {"x", "y", "both"}, default "both"
        Plane(s) in which to place the initial conditions.
    sigma_limit : float, default 10
        Largest radius [beam sizes].
    radial_step : float, default 1
        Distance between consecutive radii [beam sizes].
    angles : sequence of float, default DEFAULT_ANGLES
        Angles of the lines [deg] from the ``x_norm`` (``y_norm``) axis.

    Returns
    -------
    planes : numpy.ndarray of str, shape (N,)
        Plane of each initial condition.
    radius : numpy.ndarray, shape (N,)
        Radius of each initial condition.
    angle : numpy.ndarray, shape (N,)
        Angle of each initial condition [deg].
    z_norm : numpy.ndarray, shape (4, N)
        Normalised coordinates ``(x_norm, px_norm, y_norm, py_norm)``.

    Raises
    ------
    ValueError
        If the arguments are invalid.
    """
    if plane not in ("x", "y", "both"):
        raise ValueError(f"plane must be 'x', 'y' or 'both', got {plane!r}.")
    if radial_step <= 0:
        raise ValueError(f"radial_step must be positive, got {radial_step}.")
    if sigma_limit < radial_step:
        raise ValueError(
            f"sigma_limit ({sigma_limit}) must be at least radial_step ({radial_step})."
        )
    if len(angles) == 0:
        raise ValueError("At least one angle must be given.")
    n_radii = int(np.floor(sigma_limit / radial_step * (1 + 1e-12)))
    radii = float(radial_step) * np.arange(1, n_radii + 1)
    planes = ("x", "y") if plane == "both" else (plane,)

    labels, radius, angle, columns = [], [], [], []
    for pl in planes:
        offset = 0 if pl == "x" else 2
        for ang in angles:
            theta = np.deg2rad(ang)
            for r in radii:
                z = np.zeros(4)
                z[offset] = r * np.cos(theta)
                z[offset + 1] = r * np.sin(theta)
                labels.append(pl)
                radius.append(r)
                angle.append(float(ang))
                columns.append(z)
    return (
        np.array(labels),
        np.array(radius),
        np.array(angle),
        np.array(columns).T,
    )


def _normalise(
    coords: dict[str, np.ndarray], twiss: Any, nemitt_x: float, nemitt_y: float
) -> list[np.ndarray]:
    """
    Normalised transverse coordinates at the start of the line.

    Same transformation as ``TwissTable.get_normalized_coordinates``, for
    arrays of any shape.

    Parameters
    ----------
    coords : dict of numpy.ndarray
        Physical coordinates ``x, px, y, py, zeta, pzeta``.
    twiss : xtrack.TwissTable
        Twiss table of the line.
    nemitt_x, nemitt_y : float
        Normalised emittances [m].

    Returns
    -------
    list of 4 numpy.ndarray
        ``x_norm, px_norm, y_norm, py_norm``.
    """
    names = ("x", "px", "y", "py", "zeta", "pzeta")
    co = np.array([twiss[n][0] for n in names])
    shape = coords["x"].shape
    XX = np.array([coords[n].ravel() for n in names]) - co[:, None]
    XX_norm = np.linalg.solve(twiss.W_matrix[0], XX)
    beta0 = float(np.asarray(twiss.particle_on_co.beta0).ravel()[0])
    gamma0 = float(np.asarray(twiss.particle_on_co.gamma0).ravel()[0])
    scale = np.sqrt(np.array([nemitt_x, nemitt_x, nemitt_y, nemitt_y]) / (beta0 * gamma0))
    return [(XX_norm[i] / scale[i]).reshape(shape) for i in range(4)]


def _normal_form_coordinates(
    normal_form: NormalForm4D, z_norm: Sequence[np.ndarray]
) -> list[np.ndarray]:
    """
    Real normal form coordinates from normalised coordinates.

    Parameters
    ----------
    normal_form : NormalForm4D
        Computed normal form.
    z_norm : sequence of 4 numpy.ndarray
        ``x_norm, px_norm, y_norm, py_norm``.

    Returns
    -------
    list of 4 numpy.ndarray
        ``x_nf, px_nf, y_nf, py_nf`` (``nan`` where the input is ``nan``).
    """
    zeta = normal_form.norm_to_nf(*real_to_complex(*z_norm))
    return [np.real(v) for v in complex_to_real(*zeta)]


def track_orbits(
    line: Any,
    nemitt_x: float,
    nemitt_y: float,
    plane: Plane = "both",
    sigma_limit: float = 10.0,
    radial_step: float = 1.0,
    num_turns: int = 2048,
    normal_form: NormalForm4D | None = None,
    multithreaded: bool = False,
    method: Literal["6d", "4d"] = "6d",
    twiss: Any = None,
    angles: Sequence[float] = DEFAULT_ANGLES,
) -> Orbits:
    """
    Track orbits started along lines in normalised phase space.

    See :func:`initial_conditions` for the initial conditions. A progress
    bar is shown, updated every 10 turns.

    Parameters
    ----------
    line : xtrack.Line
        Line to track.
    nemitt_x : float
        Horizontal normalised emittance [m], defines the unit of the
        normalised coordinates.
    nemitt_y : float
        Vertical normalised emittance [m].
    plane : {"x", "y", "both"}, default "both"
        Plane(s) in which to start the particles.
    sigma_limit : float, default 10
        Largest initial radius [beam sizes].
    radial_step : float, default 1
        Distance between consecutive initial radii [beam sizes].
    num_turns : int, default 2048
        Number of turns recorded (including the initial condition).
    normal_form : NormalForm4D, optional
        Computed normal form; if given, the normal form coordinates are
        added. It must act on the normalised coordinates defined by the same
        emittances and twiss method (e.g. ``OneTurnMap4D.normal_form``).
    multithreaded : bool, default False
        Track with an OpenMP context with ``omp_num_threads="auto"``. The
        tracker of the line is rebuilt for the tracking and restored
        afterwards.
    method : {"6d", "4d"}, default "6d"
        Twiss method; with ``"4d"`` the longitudinal coordinates are frozen
        while tracking.
    twiss : xtrack.TwissTable, optional
        Twiss table of the line, computed with ``method`` if not given.
    angles : sequence of float, default DEFAULT_ANGLES
        Angles of the lines of initial conditions [deg].

    Returns
    -------
    Orbits
        The orbits.

    Raises
    ------
    ValueError
        If the arguments are invalid or the normal form is not computed.
    """
    if method not in ("6d", "4d"):
        raise ValueError(f"method must be '6d' or '4d', got {method!r}.")
    if num_turns < 1:
        raise ValueError(f"num_turns must be at least 1, got {num_turns}.")
    if nemitt_x <= 0 or nemitt_y <= 0:
        raise ValueError("The emittances nemitt_x and nemitt_y must be positive.")
    if normal_form is not None and normal_form.Psi is None:
        raise ValueError("The normal form has not been computed.")
    planes, radius, angle, z0 = initial_conditions(plane, sigma_limit, radial_step, angles)

    if twiss is None:
        twiss = line.twiss(method=method)

    original_context = line._context if multithreaded else None
    try:
        if multithreaded:
            import xobjects as xo

            line.discard_tracker()
            line.build_tracker(_context=xo.ContextCpu(omp_num_threads="auto"))
        particles = line.build_particles(
            method=method,
            x_norm=z0[0], px_norm=z0[1], y_norm=z0[2], py_norm=z0[3],
            zeta_norm=np.zeros(z0.shape[1]), pzeta_norm=np.zeros(z0.shape[1]),
            nemitt_x=nemitt_x, nemitt_y=nemitt_y,
            W_matrix=twiss.W_matrix[0], particle_on_co=twiss.particle_on_co,
        )
        kwargs = {"freeze_longitudinal": True} if method == "4d" else {}
        line.track(
            particles,
            num_turns=num_turns,
            turn_by_turn_monitor=True,
            with_progress=10,
            **kwargs,
        )
        record = line.record_last_track
        data = {
            name: np.array(getattr(record, name))
            for name in ("x", "px", "y", "py", "zeta", "pzeta", "state")
        }
    finally:
        if multithreaded:
            line.discard_tracker()
            line.build_tracker(_context=original_context)

    # Rows of the monitor are indexed by particle id, as the initial conditions
    valid = data.pop("state") > 0
    for name in data:
        data[name] = np.where(valid, data[name], np.nan)
    norm = _normalise(data, twiss, nemitt_x, nemitt_y)
    nf = (
        _normal_form_coordinates(normal_form, norm)
        if normal_form is not None
        else [None] * 4
    )
    num_valid = valid.sum(axis=1)
    return Orbits(
        plane=planes,
        radius=radius,
        angle=angle,
        lost=num_valid < num_turns,
        num_valid_turns=num_valid,
        **{n: data[n] for n in _PHYS},
        **dict(zip(_NORM, norm)),
        **dict(zip(_NF, nf)),
    )


# --------------------------------------------------------------------------- #
# Orbits from the interpolating Hamiltonian
# --------------------------------------------------------------------------- #


@dataclass
class HamiltonianOrbits:
    """
    Orbits predicted by the interpolating Hamiltonian of a normal form.

    Returned by :func:`hamiltonian_orbits`. For each particle, the orbit is
    the contour of the real Hamiltonian ``h = Re(-i H)`` through its initial
    condition, in the normal form phase space of the plane in which it was
    started (``(x_nf, px_nf)`` or ``(y_nf, py_nf)``); the normal form
    coordinates of the other plane are held at their initial values. Of the
    level set ``h = h0`` only the curves of the particle's own orbit are
    kept: the curve through the initial condition and those onto which the
    one-turn map in normal form carries it. An orbit therefore consists of
    one curve, or of one curve per island of an island chain; curves
    reaching the edge of the grid are open.

    Particles are in the same order as the initial conditions, so that they
    can be compared row by row with an :class:`Orbits` object from
    :func:`track_orbits`.

    In normal form coordinates the contours are the orbits of the plane. In
    normalised coordinates they are drawn at the initial normal form phase
    of the other plane: if the other plane has a non-zero normal form
    amplitude (for example the horizontal motion driven by the vertical one
    through sextupoles), the tracked orbit fills a band around the curve.
    The comparison in normal form coordinates is therefore the cleaner
    accuracy measure.

    Attributes
    ----------
    plane : numpy.ndarray of str, shape (num_particles,)
        Plane of each particle (``"x"`` or ``"y"``).
    radius : numpy.ndarray, shape (num_particles,)
        Initial radius in normalised phase space [beam sizes].
    angle : numpy.ndarray, shape (num_particles,)
        Initial angle in normalised phase space [deg].
    h0 : numpy.ndarray, shape (num_particles,)
        Value of ``h`` at each initial condition.
    z0_norm : numpy.ndarray, shape (4, num_particles)
        Initial normalised coordinates ``(x_norm, px_norm, y_norm, py_norm)``.
    z0_nf : numpy.ndarray, shape (4, num_particles)
        Initial normal form coordinates ``(x_nf, px_nf, y_nf, py_nf)``.
    curves_nf : list of list of numpy.ndarray
        ``curves_nf[i]`` holds the curves of particle ``i``, each an array of
        shape ``(4, n_points)`` of normal form coordinates.
    curves_norm : list of list of numpy.ndarray
        The same curves in normalised coordinates (through ``Phi``).
    normal_form : NormalForm4D
        The normal form used.
    """

    plane: np.ndarray
    radius: np.ndarray
    angle: np.ndarray
    h0: np.ndarray
    z0_norm: np.ndarray
    z0_nf: np.ndarray
    curves_nf: list[list[np.ndarray]]
    curves_norm: list[list[np.ndarray]]
    normal_form: NormalForm4D = field(repr=False)

    @property
    def num_particles(self) -> int:
        """
        Number of particles.

        Returns
        -------
        int
            Number of initial conditions.
        """
        return int(len(self.plane))

    def select(self, plane: Literal["x", "y"]) -> HamiltonianOrbits:
        """
        Orbits of the particles started in one plane.

        Parameters
        ----------
        plane : {"x", "y"}
            Plane to select.

        Returns
        -------
        HamiltonianOrbits
            The orbits of these particles only.

        Raises
        ------
        ValueError
            If ``plane`` is not ``"x"`` or ``"y"``.
        """
        if plane not in ("x", "y"):
            raise ValueError(f"plane must be 'x' or 'y', got {plane!r}.")
        idx = np.flatnonzero(self.plane == plane)
        return HamiltonianOrbits(
            plane=self.plane[idx],
            radius=self.radius[idx],
            angle=self.angle[idx],
            h0=self.h0[idx],
            z0_norm=self.z0_norm[:, idx],
            z0_nf=self.z0_nf[:, idx],
            curves_nf=[self.curves_nf[i] for i in idx],
            curves_norm=[self.curves_norm[i] for i in idx],
            normal_form=self.normal_form,
        )

    def hamiltonian(
        self,
        x_nf: np.ndarray,
        px_nf: np.ndarray,
        y_nf: np.ndarray,
        py_nf: np.ndarray,
    ) -> np.ndarray:
        """
        Real interpolating Hamiltonian ``h = Re(-i H)``.

        Parameters
        ----------
        x_nf, px_nf, y_nf, py_nf : numpy.ndarray
            Real normal form coordinates.

        Returns
        -------
        numpy.ndarray
            ``h`` at the given points.
        """
        zeta = real_to_complex(x_nf, px_nf, y_nf, py_nf)
        return np.real(-1j * self.normal_form.H.substitute(*zeta))

    def _check_orbits(self, orbits: Orbits) -> None:
        """
        Check that tracked orbits start at the same initial conditions.

        Parameters
        ----------
        orbits : Orbits
            Tracked orbits.

        Raises
        ------
        ValueError
            If the particles do not correspond.
        """
        z0 = np.array([orbits.x_norm[:, 0], orbits.px_norm[:, 0],
                       orbits.y_norm[:, 0], orbits.py_norm[:, 0]])
        if z0.shape != self.z0_norm.shape or not np.allclose(
            z0, self.z0_norm, rtol=0, atol=1e-8
        ):
            raise ValueError(
                "The tracked orbits do not start at the initial conditions of "
                "these Hamiltonian orbits."
            )

    def _nf_coordinates(self, orbits: Orbits) -> list[np.ndarray]:
        """
        Normal form coordinates of tracked orbits, with this normal form.

        Parameters
        ----------
        orbits : Orbits
            Tracked orbits.

        Returns
        -------
        list of 4 numpy.ndarray
            ``x_nf, px_nf, y_nf, py_nf`` of shape (num_particles, num_turns).
        """
        self._check_orbits(orbits)
        norm = [orbits.x_norm, orbits.px_norm, orbits.y_norm, orbits.py_norm]
        return _normal_form_coordinates(self.normal_form, norm)

    def energy_deviation(self, orbits: Orbits) -> np.ndarray:
        """
        Relative deviation of ``h`` along tracked orbits.

        A perfect normal form gives ``h`` constant along every orbit.

        Parameters
        ----------
        orbits : Orbits
            Tracked orbits started at the same initial conditions (e.g. from
            :func:`track_orbits`), in the same order.

        Returns
        -------
        numpy.ndarray, shape (num_particles, num_turns)
            ``|h(turn) - h0| / |h0|`` (``nan`` where the orbit is ``nan``).

        Raises
        ------
        ValueError
            If the orbits do not start at the same initial conditions.
        """
        h = self.hamiltonian(*self._nf_coordinates(orbits))
        return np.abs(h - self.h0[:, None]) / np.abs(self.h0)[:, None]

    def distance(
        self, orbits: Orbits, coordinates: Literal["nf", "norm"] = "nf"
    ) -> np.ndarray:
        """
        Distance of tracked orbit points from the Hamiltonian orbits.

        The distance is measured in the phase space of each particle's
        plane, from every tracked point to the nearest segment of the
        particle's curves.

        Parameters
        ----------
        orbits : Orbits
            Tracked orbits started at the same initial conditions, in the
            same order.
        coordinates : {"nf", "norm"}, default "nf"
            Compare in normal form or in normalised coordinates.

        Returns
        -------
        numpy.ndarray, shape (num_particles, num_turns)
            Distances (``nan`` where the orbit is ``nan`` or a particle has
            no curve). In normalised coordinates they include the band due
            to the motion of the other plane (see the class documentation).

        Raises
        ------
        ValueError
            If the orbits do not start at the same initial conditions or
            ``coordinates`` is invalid.
        """
        if coordinates == "nf":
            pts = self._nf_coordinates(orbits)
            curves = self.curves_nf
        elif coordinates == "norm":
            self._check_orbits(orbits)
            pts = [orbits.x_norm, orbits.px_norm, orbits.y_norm, orbits.py_norm]
            curves = self.curves_norm
        else:
            raise ValueError(f"coordinates must be 'nf' or 'norm', got {coordinates!r}.")
        out = np.full(pts[0].shape, np.nan)
        for i in range(self.num_particles):
            if not curves[i]:
                continue
            k = 0 if self.plane[i] == "x" else 2
            p = np.stack([pts[k][i], pts[k + 1][i]], axis=1)
            segments = [c[k : k + 2].T for c in curves[i]]
            out[i] = _distance_to_polylines(p, segments)
        return out


def _distance_to_polylines(points: np.ndarray, lines: list[np.ndarray]) -> np.ndarray:
    """
    Distance from points to the nearest segment of a set of polylines.

    Parameters
    ----------
    points : numpy.ndarray, shape (N, 2)
        Points.
    lines : list of numpy.ndarray, shape (M, 2)
        Polylines.

    Returns
    -------
    numpy.ndarray, shape (N,)
        Distances (``nan`` for ``nan`` points).
    """
    a = np.concatenate([ln[:-1] for ln in lines])
    b = np.concatenate([ln[1:] for ln in lines])
    if len(a) == 0:
        a = b = np.concatenate(lines)
    d = b - a
    len2 = np.maximum(np.sum(d**2, axis=1), 1e-300)
    out = np.full(len(points), np.nan)
    ok = np.all(np.isfinite(points), axis=1)
    for start in range(0, int(ok.sum()), 512):
        p = points[ok][start : start + 512]
        t = np.clip(np.einsum("pk,sk->ps", p, d) - np.sum(a * d, axis=1), 0, None) / len2
        t = np.minimum(t, 1.0)
        closest = a[None] + t[..., None] * d[None]
        dist = np.sqrt(np.min(np.sum((p[:, None] - closest) ** 2, axis=2), axis=1))
        idx = np.flatnonzero(ok)[start : start + 512]
        out[idx] = dist
    return out


def _orbit_curves(
    lines: list[np.ndarray],
    start: np.ndarray,
    one_turn: Any,
    num_probes: int = 8,
) -> list[np.ndarray]:
    """
    Curves of a level set that belong to the orbit through a point.

    A level set of ``h`` can consist of several curves belonging to
    different orbits (e.g. the core and the outside of an island chain at
    the same value of ``h``). The orbit consists of the curve through the
    initial point and of the curves onto which the one-turn map carries it
    (e.g. the other islands of a chain).

    Parameters
    ----------
    lines : list of numpy.ndarray, shape (M, 2)
        Curves of the level set in the plane.
    start : numpy.ndarray, shape (2,)
        Initial point in the plane.
    one_turn : callable
        Maps points of shape (K, 2) one turn.
    num_probes : int, default 8
        Number of points of each selected curve mapped to find the others.

    Returns
    -------
    list of numpy.ndarray
        The selected curves.
    """
    if not lines:
        return []
    first = int(np.argmin([_distance_to_polylines(start[None], [ln])[0] for ln in lines]))
    selected, todo = {first}, [first]
    while todo and len(selected) < len(lines):
        curve = lines[todo.pop()]
        probes = curve[np.linspace(0, len(curve) - 1, num_probes).astype(int)]
        images = one_turn(probes)
        dist = np.array([_distance_to_polylines(images, [ln]) for ln in lines])
        order = np.argsort(dist, axis=0)
        for k in range(images.shape[0]):
            j, second = order[0, k], order[1, k] if len(lines) > 1 else None
            # accept only unambiguous images, clearly closer to one curve
            if second is not None and not dist[j, k] < 0.5 * dist[second, k]:
                continue
            if j not in selected:
                selected.add(int(j))
                todo.append(int(j))
    return [lines[j] for j in sorted(selected)]


def _initial_planes(z0: np.ndarray, rtol: float = 1e-9) -> np.ndarray:
    """
    Plane of initial conditions given in normalised coordinates.

    Parameters
    ----------
    z0 : numpy.ndarray, shape (4, N)
        Initial normalised coordinates.
    rtol : float, default 1e-9
        The amplitude in the other plane must be below ``rtol`` times the
        amplitude in the plane (rounding errors are tolerated).

    Returns
    -------
    numpy.ndarray of str, shape (N,)
        ``"x"`` if the vertical coordinates are (nearly) zero, ``"y"`` if
        the horizontal ones are.

    Raises
    ------
    ValueError
        If a point is in neither or both planes.
    """
    r_x, r_y = np.hypot(z0[0], z0[1]), np.hypot(z0[2], z0[3])
    in_x = (r_x > 0) & (r_y <= rtol * r_x)
    in_y = (r_y > 0) & (r_x <= rtol * r_y)
    bad = ~(in_x | in_y)
    if np.any(bad):
        raise ValueError(
            "Initial conditions must lie in one plane (the normalised "
            "coordinates of the other plane zero, the point not at the "
            f"origin); {int(bad.sum())} do not."
        )
    return np.where(in_x, "x", "y")


def _hamiltonian_groups(
    normal_form: NormalForm4D, plane: str
) -> dict[tuple[int, int], list[tuple[int, int, complex]]]:
    """
    Terms of H grouped by the exponents of the other plane.

    Parameters
    ----------
    normal_form : NormalForm4D
        Computed normal form.
    plane : {"x", "y"}
        Plane of the contours.

    Returns
    -------
    dict
        ``{(m, n): [(p, q, c), ...]}`` such that
        ``H = sum c zeta^p zeta*^q w^m w*^n``, with ``(zeta, zeta*)`` the
        complex coordinates of ``plane`` and ``(w, w*)`` those of the other
        plane.
    """
    groups: dict[tuple[int, int], list[tuple[int, int, complex]]] = {}
    for t in normal_form.H.terms:
        e = t.exponents
        own, other = ((e[0], e[1]), (e[2], e[3])) if plane == "x" else ((e[2], e[3]), (e[0], e[1]))
        groups.setdefault(other, []).append((own[0], own[1], complex(t.coeff)))
    return groups


def hamiltonian_orbits(
    normal_form: NormalForm4D,
    initial: Orbits | np.ndarray,
    grid_size: int = 401,
    margin: float = 1.5,
    extent: float | None = None,
) -> HamiltonianOrbits:
    """
    Orbits predicted by the interpolating Hamiltonian of a normal form.

    Each particle's orbit is the contour, through its initial condition, of
    the real Hamiltonian ``h = Re(-i H)`` in the normal form phase space of
    its plane, with the normal form coordinates of the other plane held at
    their initial values. The contours are mapped back to normalised
    coordinates with ``Phi``. See :class:`HamiltonianOrbits`.

    This section is exactly invariant when ``H`` depends on the other plane
    only through its action (non-resonant normal forms, resonances of one
    plane only). Otherwise a warning reports the relative size of the terms
    coupling the planes.

    ``h`` is an invariant of the one-turn map for non-resonant and
    quasi-resonant normal forms. Near a resonance (e.g. to describe resonance
    islands) use ``res_case="quasiresonant"`` with the resonant eigenvalues:
    with ``res_case="resonant"`` the map is ``Lambda o flow(U_nonlinear)``
    and ``h`` is invariant only if the tunes are exactly resonant.

    Parameters
    ----------
    normal_form : NormalForm4D
        Computed normal form, acting on the same normalised coordinates as
        the initial conditions.
    initial : Orbits or array_like, shape (4, N)
        Initial conditions: tracked orbits (their first turn and labels are
        used) or normalised coordinates ``(x_norm, px_norm, y_norm, py_norm)``
        of points lying in one plane each.
    grid_size : int, default 401
        Number of grid points per axis used to compute the contours.
    margin : float, default 1.5
        Half-width of the grid, relative to the largest initial normal form
        radius of the plane.
    extent : float, optional
        Half-width of the grid in normal form coordinates, overriding
        ``margin``.

    Returns
    -------
    HamiltonianOrbits
        The orbits.

    Raises
    ------
    ValueError
        If the normal form is not computed or the initial conditions are
        invalid.
    """
    if normal_form.H is None or normal_form.Psi is None:
        raise ValueError("The normal form has not been computed.")
    if grid_size < 3:
        raise ValueError(f"grid_size must be at least 3, got {grid_size}.")
    if isinstance(initial, Orbits):
        z0 = np.array([initial.x_norm[:, 0], initial.px_norm[:, 0],
                       initial.y_norm[:, 0], initial.py_norm[:, 0]])
        planes, radius, angle = initial.plane, initial.radius, initial.angle
    else:
        z0 = np.atleast_2d(np.asarray(initial, dtype=float))
        if z0.ndim != 2 or z0.shape[0] != 4:
            raise ValueError(f"initial must have shape (4, N), got {z0.shape}.")
        planes = _initial_planes(z0)
        k = np.where(planes == "x", 0, 2)
        cols = np.arange(z0.shape[1])
        a, b = z0[k, cols], z0[k + 1, cols]
        radius, angle = np.hypot(a, b), np.rad2deg(np.arctan2(b, a))

    zeta0 = normal_form.norm_to_nf(*real_to_complex(*z0))
    z0_nf = np.real(np.array(complex_to_real(*zeta0)))
    zeta0 = np.array(zeta0)
    h0 = np.real(-1j * normal_form.H.substitute(*zeta0))

    n = z0.shape[1]
    curves_nf: list[list[np.ndarray]] = [[] for _ in range(n)]
    curves_norm: list[list[np.ndarray]] = [[] for _ in range(n)]
    for plane in ("x", "y"):
        idx = np.flatnonzero(planes == plane)
        if idx.size == 0:
            continue
        own = (0, 1) if plane == "x" else (2, 3)
        oth = (2, 3) if plane == "x" else (0, 1)
        half = extent if extent is not None else margin * np.max(np.abs(zeta0[own[0], idx]))
        axis = np.linspace(-half, half, grid_size)
        A, B = np.meshgrid(axis, axis, indexing="xy")
        u, v = A - 1j * B, A + 1j * B
        groups = _hamiltonian_groups(normal_form, plane)
        max_exp = max(max(p, q) for terms in groups.values() for p, q, _ in terms)
        u_pow = [np.ones_like(u)]
        v_pow = [np.ones_like(v)]
        for _ in range(max_exp):
            u_pow.append(u_pow[-1] * u)
            v_pow.append(v_pow[-1] * v)
        grids = {
            other: sum(c * u_pow[p] * v_pow[q] for p, q, c in terms)
            for other, terms in groups.items()
        }
        coupling_ratio = 0.0
        for i in idx:
            w, ws = zeta0[oth[0], i], zeta0[oth[1], i]
            parts = {other: g * w ** other[0] * ws ** other[1] for other, g in grids.items()}
            H_grid = sum(parts.values())
            h_grid = np.real(-1j * H_grid)
            coupling = [g for (m, nn), g in parts.items() if m != nn]
            span = np.ptp(h_grid)
            if coupling and span > 0:
                ratio = float(np.max(np.abs(sum(coupling)))) / span
                coupling_ratio = max(coupling_ratio, ratio)
            lines = contourpy.contour_generator(x=A, y=B, z=h_grid).lines(h0[i])
            lines = [ln for ln in lines if len(ln) >= 2]

            def one_turn(points: np.ndarray, i: int = i) -> np.ndarray:
                full = np.empty((4, len(points)))
                full[own[0]], full[own[1]] = points[:, 0], points[:, 1]
                full[oth[0]], full[oth[1]] = z0_nf[oth[0], i], z0_nf[oth[1], i]
                image = complex_to_real(*normal_form.N.substitute(*real_to_complex(*full)))
                return np.real(np.array([image[own[0]], image[own[1]]])).T

            start = z0_nf[[own[0], own[1]], i]
            for line in _orbit_curves(lines, start, one_turn):
                pts = np.empty((4, len(line)))
                pts[own[0]], pts[own[1]] = line[:, 0], line[:, 1]
                pts[oth[0]], pts[oth[1]] = z0_nf[oth[0], i], z0_nf[oth[1], i]
                curves_nf[i].append(pts)
                norm = complex_to_real(*normal_form.nf_to_norm(*real_to_complex(*pts)))
                curves_norm[i].append(np.real(np.array(norm)))
        if coupling_ratio > 1e-6:
            warnings.warn(
                f"The Hamiltonian couples the planes: in the {plane} plane the "
                f"coupling terms reach {coupling_ratio:.1e} of the variation of h "
                "on the grid, so the contours are only approximately invariant.",
                stacklevel=2,
            )
    return HamiltonianOrbits(
        plane=np.asarray(planes),
        radius=np.asarray(radius, dtype=float),
        angle=np.asarray(angle, dtype=float),
        h0=h0,
        z0_norm=z0,
        z0_nf=z0_nf,
        curves_nf=curves_nf,
        curves_norm=curves_norm,
        normal_form=normal_form,
    )


# --------------------------------------------------------------------------- #
# Invariant tori of the normal form Hamiltonian
# --------------------------------------------------------------------------- #

#: Terms of H smaller than this fraction of its largest term are ignored when
#: identifying the resonance (rounding noise).
_RES_TOL = 1e-12
#: Accuracy of the inversion of Psi for the points of the tori [beam sizes].
_TORI_TOL = 1e-10
#: Newton steps refining the points of the reduced curves onto the level set.
_REFINE_STEPS = 4
#: A reduced curve spanning fewer grid cells than this is not resolved.
_MIN_CELLS = 10
#: Factor by which successive zoomed grids of an unresolved reduced curve
#: shrink.
_ZOOM = 4.0
#: Smallest zoomed grid half-width, relative to the initial amplitude of the
#: chart plane (the start must lie well inside the grid).
_ZOOM_FLOOR = 1.5


@dataclass
class HamiltonianTori:
    """
    Invariant tori of the interpolating Hamiltonian of a normal form.

    Returned by :func:`hamiltonian_tori`. For each initial condition, the
    torus is the set of points of the 4D phase space that the normal form
    predicts its orbit to fill. It is given as arrays of points
    ``(4, M, T)``: along axis 1 the points follow a curve of the reduced
    motion (see :func:`hamiltonian_tori`), along axis 2 the phase that does
    not appear in ``H``. An orbit consists of one torus, or of
    several for the islands of a resonance.

    To compare with tracking in the plane of a particle, plot all points
    (the projection of the torus is a band when the planes are coupled), or
    the curves at fixed values of the free phase::

        for t in tori.tori_norm[i]:
            ax.plot(t[0].ravel(), t[1].ravel(), ",")       # band in (x, px)
            ax.plot(t[0][:, 0], t[1][:, 0])                # one curve

    Attributes
    ----------
    plane : numpy.ndarray of str, shape (num_particles,)
        Plane of each initial condition (that of the larger normalised
        amplitude).
    resonance : tuple of 2 int or None
        Resonance ``(n, m)`` on which ``H`` depends through the phase
        ``n theta1 + m theta2``, or None if ``H`` depends on the actions only.
    h0 : numpy.ndarray, shape (num_particles,)
        Value of ``h = Re(-i H)`` at each initial condition.
    invariant : numpy.ndarray, shape (num_particles,)
        Second invariant ``m |zeta1|^2 - n |zeta2|^2`` (``nan`` if
        ``resonance`` is None, when ``|zeta1|`` and ``|zeta2|`` are both
        invariant).
    z0_norm, z0_nf : numpy.ndarray, shape (4, num_particles)
        Initial normalised and normal form coordinates.
    tori_nf, tori_norm : list of list of numpy.ndarray
        ``tori_nf[i]`` holds the tori of particle ``i``, each an array of
        shape ``(4, M, T)`` of normal form coordinates;
        ``tori_norm`` the same in normalised coordinates (``nan`` where
        ``Psi`` could not be inverted).
    normal_form : NormalForm4D or list of NormalForm4D
        The normal form used, or the normal forms joined by seams.
    region : numpy.ndarray of int, shape (num_particles,) or None
        With normal forms joined by seams, the index of the normal form used
        for each particle.
    """

    plane: np.ndarray
    resonance: tuple[int, int] | None
    h0: np.ndarray
    invariant: np.ndarray
    z0_norm: np.ndarray
    z0_nf: np.ndarray
    tori_nf: list[list[np.ndarray]]
    tori_norm: list[list[np.ndarray]]
    normal_form: NormalForm4D | list[NormalForm4D] = field(repr=False)
    region: np.ndarray | None = None

    @property
    def num_particles(self) -> int:
        """
        Number of particles.

        Returns
        -------
        int
            Number of initial conditions.
        """
        return int(len(self.plane))

    def select(self, plane: Literal["x", "y"]) -> HamiltonianTori:
        """
        Tori of the particles started in one plane.

        The particles keep their order, so the result lines up row by row
        with ``Orbits.select(plane)`` of the orbits the tori were computed
        from.

        Parameters
        ----------
        plane : {"x", "y"}
            Plane to select.

        Returns
        -------
        HamiltonianTori
            The tori of these particles only.

        Raises
        ------
        ValueError
            If ``plane`` is not ``"x"`` or ``"y"``.
        """
        if plane not in ("x", "y"):
            raise ValueError(f"plane must be 'x' or 'y', got {plane!r}.")
        idx = np.flatnonzero(self.plane == plane)
        return HamiltonianTori(
            plane=self.plane[idx],
            resonance=self.resonance,
            h0=self.h0[idx],
            invariant=self.invariant[idx],
            z0_norm=self.z0_norm[:, idx],
            z0_nf=self.z0_nf[:, idx],
            tori_nf=[self.tori_nf[i] for i in idx],
            tori_norm=[self.tori_norm[i] for i in idx],
            normal_form=self.normal_form,
            region=None if self.region is None else self.region[idx],
        )

    def points(self, i: int, coordinates: Literal["nf", "norm"] = "norm") -> np.ndarray:
        """
        All points of the tori of one particle.

        Parameters
        ----------
        i : int
            Particle.
        coordinates : {"nf", "norm"}, default "norm"
            Normal form or normalised coordinates.

        Returns
        -------
        numpy.ndarray, shape (4, M)
            Points (those that could not be computed are left out).

        Raises
        ------
        ValueError
            If ``coordinates`` is invalid.
        """
        if coordinates not in ("nf", "norm"):
            raise ValueError(f"coordinates must be 'nf' or 'norm', got {coordinates!r}.")
        tori = self.tori_nf[i] if coordinates == "nf" else self.tori_norm[i]
        if not tori:
            return np.empty((4, 0))
        pts = np.concatenate([t.reshape(4, -1) for t in tori], axis=1)
        return np.real(pts[:, np.all(np.isfinite(pts), axis=0)])

    def distance(
        self, orbits: Orbits, coordinates: Literal["nf", "norm"] = "norm"
    ) -> np.ndarray:
        """
        4D distance of tracked orbit points from the tori.

        The distance is that to the nearest point of the particle's tori, so
        it includes the spacing of the points (about
        ``2 pi r / sqrt(num_points)`` for a torus of radius ``r``).

        Parameters
        ----------
        orbits : Orbits
            Tracked orbits started at the same initial conditions, in the
            same order; with normal form coordinates for ``coordinates="nf"``.
        coordinates : {"nf", "norm"}, default "norm"
            Compare in normal form or in normalised coordinates.

        Returns
        -------
        numpy.ndarray, shape (num_particles, num_turns)
            Distances [beam sizes] (``nan`` where the orbit is ``nan`` or a
            particle has no torus).

        Raises
        ------
        ValueError
            If the orbits do not start at the same initial conditions, lack
            normal form coordinates, or ``coordinates`` is invalid.
        """
        from scipy.spatial import cKDTree

        z0 = np.array([orbits.x_norm[:, 0], orbits.px_norm[:, 0],
                       orbits.y_norm[:, 0], orbits.py_norm[:, 0]])
        if z0.shape != self.z0_norm.shape or not np.allclose(z0, self.z0_norm, atol=1e-9):
            raise ValueError("The orbits do not start at the initial conditions of the tori.")
        names = _NORM if coordinates == "norm" else _NF
        if coordinates == "nf" and not orbits.has_normal_form:
            raise ValueError("The orbits have no normal form coordinates.")
        pts = np.array([getattr(orbits, c) for c in names])  # (4, N, T)
        out = np.full(pts.shape[1:], np.nan)
        for i in range(self.num_particles):
            torus = self.points(i, coordinates)
            if torus.shape[1] == 0:
                continue
            p = pts[:, i].T
            ok = np.all(np.isfinite(p), axis=1)
            out[i, ok] = cKDTree(torus.T).query(p[ok])[0]
        return out


def _resonance_of(H: Any) -> tuple[int, int] | None:
    """
    Resonance on which a normal form Hamiltonian depends.

    Every term ``c zeta1^a zeta1*^b zeta2^c zeta2*^d`` depends on the phases
    through ``(a - b) theta1 + (c - d) theta2``. In a normal form with a
    single resonance, all the vectors ``(a - b, c - d)`` are multiples of the
    resonance ``(n, m)``.

    Parameters
    ----------
    H : Polynom
        Hamiltonian.

    Returns
    -------
    tuple of 2 int or None
        Primitive resonance ``(n, m)`` (first non-zero entry positive), or
        None if ``H`` depends on the actions only.

    Raises
    ------
    ValueError
        If ``H`` depends on two independent combinations of the phases.
    """
    terms = [(t.exponents, complex(t.coeff)) for t in H.terms]
    scale = max((abs(c) for _, c in terms), default=0.0)
    vectors = {
        (e[0] - e[1], e[2] - e[3])
        for e, c in terms
        if abs(c) > _RES_TOL * scale and (e[0] != e[1] or e[2] != e[3])
    }
    if not vectors:
        return None
    p, q = next(iter(vectors))
    g = np.gcd(abs(p), abs(q))
    n, m = p // g, q // g
    if n < 0 or (n == 0 and m < 0):
        n, m = -n, -m
    if any(a * m - b * n != 0 for a, b in vectors):
        raise ValueError(
            "The Hamiltonian depends on two independent combinations of the "
            "phases (a double resonance): its motion is not integrable and has "
            "no invariant tori."
        )
    return int(n), int(m)


def _resample(line: np.ndarray, num: int) -> np.ndarray:
    """
    Points equally spaced along a polyline.

    Parameters
    ----------
    line : numpy.ndarray, shape (M, 2)
        Polyline (closed if its ends coincide).
    num : int
        Number of points.

    Returns
    -------
    numpy.ndarray, shape (num, 2)
        Points (for a closed polyline the last one is not repeated).
    """
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(line, axis=0).T))])
    closed = np.allclose(line[0], line[-1])
    t = np.linspace(0, s[-1], num, endpoint=not closed)
    return np.stack([np.interp(t, s, line[:, 0]), np.interp(t, s, line[:, 1])], axis=1)


def hamiltonian_tori(
    normal_form: NormalForm4D | Sequence[NormalForm4D],
    initial: Orbits | np.ndarray,
    num_points: int = 32768,
    grid_size: int = 401,
    margin: float = 1.5,
    seams: Sequence[float] | None = None,
) -> HamiltonianTori:
    """
    Invariant tori of the interpolating Hamiltonian through initial points.

    With a single normal form, see :func:`_hamiltonian_tori_single` for the
    method. Several normal forms (2 to 4, e.g. from one-turn maps fitted on
    different amplitude ranges, all keeping the same resonance) can be joined
    at ``seams`` of the normal form amplitude
    ``rho = sqrt(|zeta1|^2 + |zeta2|^2)``, as for the distributions of
    :mod:`xnf4d.distributions`: each initial condition ``z0`` gets its torus
    from a normal form ``k`` with ``rho_k(z0)`` between ``seams[k - 1]`` and
    ``seams[k]``. Far outside its fit range a normal form can place a point
    in its band by accident, so among several such normal forms the one
    whose band is closest to the normalised amplitude ``|z0|`` is used; if
    there is none, the normal form of the band of ``|z0|``.

    Parameters
    ----------
    normal_form : NormalForm4D or sequence of NormalForm4D
        Computed normal form, or normal forms from the innermost.
    initial : Orbits or array_like, shape (4, N)
        Initial conditions (see :func:`_hamiltonian_tori_single`).
    num_points, grid_size, margin
        See :func:`_hamiltonian_tori_single`.
    seams : sequence of float, optional
        Increasing seam radii [beam sizes], one fewer than the normal forms.

    Returns
    -------
    HamiltonianTori
        The tori (with :attr:`HamiltonianTori.region` for several normal
        forms).

    Raises
    ------
    ValueError
        If a normal form is not computed, has two resonances, the normal
        forms keep different resonances, or the seams are invalid.
    """
    if not isinstance(normal_form, Sequence):
        if seams is not None and len(seams) > 0:
            raise ValueError("seams require a sequence of normal forms.")
        return _hamiltonian_tori_single(normal_form, initial, num_points, grid_size, margin)
    forms = list(normal_form)
    seams = np.asarray([] if seams is None else seams, dtype=float)
    if not 2 <= len(forms) <= 4:
        raise ValueError(f"Between 2 and 4 normal forms can be joined, got {len(forms)}.")
    if seams.ndim != 1 or seams.size != len(forms) - 1:
        raise ValueError(
            f"{len(forms)} normal forms need {len(forms) - 1} seam radii, got {seams.tolist()}."
        )
    if np.any(seams <= 0) or np.any(np.diff(seams) <= 0):
        raise ValueError(f"The seam radii must be positive and increasing, got {seams.tolist()}.")
    for nf in forms:
        if nf.H is None or nf.Psi is None:
            raise ValueError("The normal form has not been computed.")
    resonances = {_resonance_of(nf.H) for nf in forms}
    if len(resonances) > 1:
        raise ValueError(
            f"The normal forms keep different resonances {sorted(map(str, resonances))}; "
            "their tori cannot be joined."
        )
    z0, planes = _initial_points(initial)
    # band of each normal form, and amplitude of the points in each
    lower = np.concatenate([[0.0], seams])
    upper = np.concatenate([seams, [np.inf]])
    rho = np.array([
        np.sqrt(np.sum(np.abs(np.array(nf.norm_to_nf(*real_to_complex(*z0)))[[0, 2]]) ** 2, axis=0))
        for nf in forms
    ])
    inside = (rho >= lower[:, None]) & (rho < upper[:, None])
    # Far outside its fit range a normal form can map a point to a radius in
    # its own band by accident: among the normal forms placing the point in
    # their band, take the one whose band is closest to its normalised
    # amplitude; if none does, the band of the normalised amplitude.
    r_lin = np.sqrt(np.sum(z0**2, axis=0))
    off_band = np.maximum(lower[:, None] - r_lin, r_lin - upper[:, None]).clip(min=0)
    region = np.where(
        inside.any(axis=0),
        np.argmin(np.where(inside, off_band, np.inf), axis=0),
        np.searchsorted(seams, r_lin, side="right"),
    )
    n = z0.shape[1]
    parts = {}
    for k, nf in enumerate(forms):
        idx = np.flatnonzero(region == k)
        if idx.size:
            parts[k] = (idx, _hamiltonian_tori_single(nf, z0[:, idx], num_points, grid_size, margin))
    h0, invariant, z0_nf = np.empty(n), np.empty(n), np.empty((4, n))
    tori_nf: list[list[np.ndarray]] = [[] for _ in range(n)]
    tori_norm: list[list[np.ndarray]] = [[] for _ in range(n)]
    for idx, part in parts.values():
        h0[idx], invariant[idx], z0_nf[:, idx] = part.h0, part.invariant, part.z0_nf
        for j, i in enumerate(idx):
            tori_nf[i], tori_norm[i] = part.tori_nf[j], part.tori_norm[j]
    return HamiltonianTori(
        plane=planes, resonance=resonances.pop(), h0=h0, invariant=invariant,
        z0_norm=z0, z0_nf=z0_nf, tori_nf=tori_nf, tori_norm=tori_norm,
        normal_form=forms, region=region,
    )


def _initial_points(initial: Orbits | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Initial normalised coordinates and planes of initial conditions.

    Parameters
    ----------
    initial : Orbits or array_like, shape (4, N)
        Tracked orbits (their first turn and labels are used) or normalised
        coordinates.

    Returns
    -------
    z0 : numpy.ndarray, shape (4, N)
        Normalised coordinates.
    planes : numpy.ndarray of str, shape (N,)
        Plane of each point (the labels of the orbits, or the plane of the
        larger normalised amplitude).

    Raises
    ------
    ValueError
        If the array has the wrong shape.
    """
    if isinstance(initial, Orbits):
        z0 = np.array([initial.x_norm[:, 0], initial.px_norm[:, 0],
                       initial.y_norm[:, 0], initial.py_norm[:, 0]])
        return z0, np.asarray(initial.plane)
    z0 = np.atleast_2d(np.asarray(initial, dtype=float))
    if z0.ndim != 2 or z0.shape[0] != 4:
        raise ValueError(f"initial must have shape (4, N), got {z0.shape}.")
    return z0, np.where(np.hypot(z0[0], z0[1]) >= np.hypot(z0[2], z0[3]), "x", "y")


def _hamiltonian_tori_single(
    normal_form: NormalForm4D,
    initial: Orbits | np.ndarray,
    num_points: int = 32768,
    grid_size: int = 401,
    margin: float = 1.5,
) -> HamiltonianTori:
    """
    Invariant tori of the interpolating Hamiltonian of one normal form.

    Unlike :func:`hamiltonian_orbits`, which draws the contour of ``h`` in one
    plane with the other plane frozen, this follows the 4D motion, in which
    the planes can exchange energy (e.g. near a coupling resonance).

    ``h = Re(-i H)`` depends on the actions ``A1 = |zeta1|^2``,
    ``A2 = |zeta2|^2`` and, in a normal form with one resonance ``(n, m)``
    (given by ``res_basis1``), on the phase ``phi = n theta1 + m theta2``. It
    is invariant under the shifts of the phases along ``(m, -n)``, so
    ``K = m A1 - n A2`` is a second invariant (``A1 + A2`` for the coupling
    resonance ``(1, -1)``). At fixed ``K`` the motion reduces to one degree
    of freedom, whose orbits are the level curves of ``h``; the torus is such
    a curve swept along the free phase. Steps:

    1. ``zeta0 = Psi(z0)`` gives ``h0`` and ``K``.
    2. The reduced curve is the contour ``h = h0`` in the complex plane
       ``zeta_j`` of one oscillator (``j`` of the plane whose phase enters
       ``phi``, of smaller action), the other one having its action set by
       ``K`` and its phase fixed at the initial value. In this chart the
       state with ``A_j = 0`` is a regular point (the origin), which in polar
       coordinates ``(phi, A_j)`` it is not. The contour is computed on a grid
       and its points refined by Newton's method onto ``h = h0``. The curves
       of the orbit are selected as in :func:`hamiltonian_orbits` (several for
       an island chain).
    3. Each point is swept along the free phase ``(m, -n)``.
    4. The points are mapped to normalised coordinates by solving
       ``Psi(z) = zeta`` with Newton's method (consistent with the normal form
       coordinates ``Psi(z)`` of tracked particles).

    If ``H`` depends on the actions only (non-resonant normal form), the tori
    are those of constant ``A1`` and ``A2``, swept along both phases (axis 1
    the phase of the particle's plane). A normal form with two resonances
    has no invariant tori and is rejected.

    ``h`` and ``K`` are invariants of the one-turn map for non-resonant and
    quasi-resonant normal forms (see :func:`hamiltonian_orbits`).

    Parameters
    ----------
    normal_form : NormalForm4D
        Computed normal form, acting on the same normalised coordinates as
        the initial conditions.
    initial : Orbits or array_like, shape (4, N)
        Initial conditions: tracked orbits (their first turn and labels are
        used) or normalised coordinates ``(x_norm, px_norm, y_norm, py_norm)``
        (any 4D points).
    num_points : int, default 32768
        Approximate number of points of the tori of an orbit, divided between
        the two directions of the tori (along the reduced curves and along
        the free phase) so that the points are equally spaced in both.
    grid_size : int, default 401
        Number of grid points per axis of the contour of the reduced motion.
    margin : float, default 1.5
        Half-width of the grid, relative to the initial amplitude of the
        chart plane, when the reduced motion is not bounded by ``K``
        (otherwise the grid covers all the allowed actions).

    Returns
    -------
    HamiltonianTori
        The tori.

    Raises
    ------
    ValueError
        If the normal form is not computed, has two resonances, or the
        arguments are invalid.

    Warns
    -----
    UserWarning
        If no contour is found for some initial conditions, or ``Psi`` could
        not be inverted at some points of the tori.
    """
    from .distributions import _PolyEvaluator, _to_complex, _Transforms

    if normal_form.H is None or normal_form.Psi is None:
        raise ValueError("The normal form has not been computed.")
    if grid_size < 3 or num_points < 64:
        raise ValueError("grid_size >= 3 and num_points >= 64 are required.")
    z0, planes = _initial_points(initial)

    resonance = _resonance_of(normal_form.H)
    evaluate_h = _PolyEvaluator([normal_form.H])
    transforms = _Transforms(normal_form, "x")

    def h_of(zeta: np.ndarray) -> np.ndarray:
        return np.real(-1j * evaluate_h(zeta)[0])

    zeta0 = np.array(normal_form.norm_to_nf(*real_to_complex(*z0)), dtype=complex)
    z0_nf = np.real(np.array(complex_to_real(*zeta0)))
    h0 = h_of(zeta0)
    amp0 = np.abs(zeta0[[0, 2]]) ** 2  # (2, N): A1, A2
    th0 = np.angle(zeta0[[0, 2]])
    if resonance is None:
        invariant = np.full(z0.shape[1], np.nan)
    else:
        n, m = resonance
        invariant = m * amp0[0] - n * amp0[1]

    tori_nf: list[list[np.ndarray]] = []
    n_empty = 0
    for i in range(z0.shape[1]):
        r = np.sqrt(amp0[:, i])
        if resonance is None:
            # tori of constant actions; axis 1 the phase of the particle's plane
            own = 0 if planes[i] == "x" else 1
            n_a, n_b = _split_points(num_points, 2 * np.pi * r[own], 2 * np.pi * r[1 - own])
            a = np.linspace(0, 2 * np.pi, n_a, endpoint=False)[:, None]
            b = np.linspace(0, 2 * np.pi, n_b, endpoint=False)[None, :]
            th = [None, None]
            th[own] = th0[own, i] + a + 0 * b
            th[1 - own] = th0[1 - own, i] + 0 * a + b
            tori_nf.append([_torus_from_phases(r[0], r[1], th[0], th[1])])
            continue
        curves = _reduced_curves(
            normal_form, h_of, resonance, amp0[:, i], th0[:, i], h0[i], invariant[i],
            grid_size, margin,
        )
        if not curves:
            n_empty += 1
            tori_nf.append([])
            continue
        j, w, curve_list, g, step = curves

        def chart_to_torus(p: np.ndarray, tau: np.ndarray) -> np.ndarray:
            # points (M, 2) of the chart swept along the free phase tau (1, T)
            a_j = np.sum(p**2, axis=1)
            a_o = _other_action(resonance, j, a_j, invariant[i])
            th_j = np.arctan2(p[:, 1], p[:, 0])[:, None] + tau * w[j]
            th_o = th0[1 - j, i] + tau * w[1 - j] + 0 * th_j
            r_j, r_o = np.sqrt(a_j)[:, None], np.sqrt(np.maximum(a_o, 0.0))[:, None]
            if j == 0:
                return _torus_from_phases(r_j, r_o, th_j, th_o)
            return _torus_from_phases(r_o, r_j, th_o, th_j)

        # 4D lengths of the reduced curves and of the loops of the free phase,
        # to space the points equally along both directions of the tori
        zero = np.zeros((1, 1))
        lengths, loops = [], []
        for c in curve_list:
            pts = np.real(np.array(complex_to_real(*chart_to_torus(c, zero)[:, :, 0])))
            lengths.append(np.sum(np.linalg.norm(np.diff(pts, axis=1), axis=0)))
            loops.append(2 * np.pi * np.mean(np.sqrt(w[0] ** 2 * (pts[0] ** 2 + pts[1] ** 2)
                                                     + w[1] ** 2 * (pts[2] ** 2 + pts[3] ** 2))))
        lengths = np.array(lengths)
        if lengths.sum() > 0:
            n_s, n_t = _split_points(num_points, lengths.sum(), float(np.mean(loops)))
            counts = np.maximum(8, np.round(n_s * lengths / lengths.sum())).astype(int)
        else:  # reduced curve below resolution: a single point
            n_t, counts = num_points, [1]
        tau = np.linspace(0, 2 * np.pi, n_t, endpoint=False)[None, :]
        tori = []
        for c, k in zip(curve_list, counts):
            # resample, then refine onto h = h0 (interpolation leaves the level)
            p = c if len(c) == 1 else _refine_onto_level(_resample(c, int(k)), g, h0[i], step)
            tori.append(chart_to_torus(p, tau))
        tori_nf.append(tori)

    tori_norm: list[list[np.ndarray]] = []
    n_failed = n_total = 0
    for tori in tori_nf:
        out = []
        for t in tori:
            z = transforms.psi_inverse(t, _TORI_TOL)
            n_failed += int(np.sum(~np.all(np.isfinite(z), axis=0)))
            n_total += int(np.prod(z.shape[1:]))
            out.append(z)
        tori_norm.append(out)
    tori_nf = [[np.real(np.array(complex_to_real(*t))) for t in tori] for tori in tori_nf]
    if n_empty:
        warnings.warn(
            f"No contour of the reduced motion was found for {n_empty} initial "
            "conditions (outside the grid?); their tori are empty.",
            stacklevel=2,
        )
    if n_failed:
        warnings.warn(
            f"Psi could not be inverted at {n_failed} of {n_total} points of the "
            "tori (the normal form is not valid there); they are nan.",
            stacklevel=2,
        )
    return HamiltonianTori(
        plane=np.asarray(planes), resonance=resonance, h0=h0, invariant=invariant,
        z0_norm=z0, z0_nf=z0_nf, tori_nf=tori_nf, tori_norm=tori_norm,
        normal_form=normal_form,
    )


def _split_points(total: int, length_a: float, length_b: float) -> tuple[int, int]:
    """
    Numbers of points along the two directions of a torus.

    Parameters
    ----------
    total : int
        Approximate total number of points.
    length_a, length_b : float
        Lengths of the torus along the two directions.

    Returns
    -------
    n_a, n_b : int
        Numbers of points, with ``n_a n_b`` about ``total`` and spacings
        ``length_a / n_a`` and ``length_b / n_b`` about equal (at least 8
        points along ``a`` and 1 along ``b``).
    """
    if not length_b > 0:
        return total, 1
    if not length_a > 0:
        return 8, max(1, total // 8)
    n_a = int(np.clip(round(np.sqrt(total * length_a / length_b)), 8, total))
    return n_a, max(1, round(total / n_a))


def _torus_from_phases(
    r1: np.ndarray, r2: np.ndarray, th1: np.ndarray, th2: np.ndarray
) -> np.ndarray:
    """
    Complex normal form coordinates from amplitudes and phases.

    Parameters
    ----------
    r1, r2 : array_like
        Amplitudes ``|zeta1|``, ``|zeta2|`` (broadcast with the phases).
    th1, th2 : array_like
        Phases.

    Returns
    -------
    numpy.ndarray, shape (4, *S)
        ``(zeta1, zeta1*, zeta2, zeta2*)``.
    """
    a = np.asarray(r1) * np.exp(1j * np.asarray(th1))
    b = np.asarray(r2) * np.exp(1j * np.asarray(th2))
    a, b = np.broadcast_arrays(a, b)
    return np.array([a, np.conj(a), b, np.conj(b)])


def _other_action(
    resonance: tuple[int, int], j: int, a_j: np.ndarray, invariant: float
) -> np.ndarray:
    """
    Action of the other oscillator from ``K = m A1 - n A2``.

    Parameters
    ----------
    resonance : tuple of 2 int
        ``(n, m)``.
    j : {0, 1}
        Oscillator whose action ``a_j`` is given (its coefficient non-zero).
    a_j : numpy.ndarray
        Action ``|zeta_j|^2``.
    invariant : float
        ``K``.

    Returns
    -------
    numpy.ndarray
        Action of the other oscillator (negative where not allowed).
    """
    n, m = resonance
    if j == 0:
        return (m * a_j - invariant) / n
    return (invariant + n * a_j) / m


def _reduced_curves(
    normal_form: NormalForm4D,
    h_of: Any,
    resonance: tuple[int, int],
    amp0: np.ndarray,
    th0: np.ndarray,
    h0: float,
    invariant: float,
    grid_size: int,
    margin: float,
) -> tuple[int, np.ndarray, list[np.ndarray]] | None:
    """
    Curves of the reduced motion through an initial point.

    The reduced motion at fixed ``K`` is described in the complex plane
    ``zeta_j`` of one oscillator (the other one with action set by ``K`` and
    phase fixed at its initial value), where its orbits are the contours of
    ``h``.

    Parameters
    ----------
    normal_form : NormalForm4D
        Normal form (its map ``N`` selects the curves of an island chain).
    h_of : callable
        ``h`` of complex normal form coordinates, shape ``(4, *S)``.
    resonance : tuple of 2 int
        ``(n, m)``.
    amp0, th0 : numpy.ndarray, shape (2,)
        Initial actions ``|zeta|^2`` and phases of the two oscillators.
    h0 : float
        Initial value of ``h``.
    invariant : float
        ``K``.
    grid_size : int
        Number of grid points per axis.
    margin : float
        Half-width of the grid relative to the initial amplitude of the chart
        plane, if the reduced motion is unbounded.

    Returns
    -------
    tuple or None
        ``(j, w, curves, g, step)``: chart oscillator, free phase direction
        ``(m, -n)``, curves (arrays of shape ``(M, 2)`` of
        ``(Re zeta_j, Im zeta_j)`` on the grid), ``h`` in the chart
        ``g(Re zeta_j, Im zeta_j)`` and the grid spacing; None if no contour
        is found.
    """
    n, m = resonance
    w = np.array([m, -n])
    coef = (n, m)
    candidates = [k for k in (0, 1) if coef[k] != 0]
    j = min(candidates, key=lambda k: amp0[k])
    o = 1 - j
    if n * m < 0:  # bounded: the other action vanishes at the edge
        a_max = invariant / m if j == 0 else -invariant / n
        half = np.sqrt(max(a_max, 0.0)) * (1 + 1e-9)
    else:
        half = margin * np.sqrt(max(amp0[j], 1e-12 * max(amp0.sum(), 1e-30)))
    if not half > 0:
        return None

    def zeta_at(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a_j = u * u + v * v
        a_o = _other_action(resonance, j, a_j, invariant)
        zj = u + 1j * v
        zo = np.sqrt(np.where(a_o >= 0, a_o, np.nan)) * np.exp(1j * th0[o])
        zeta = np.array([zj, np.conj(zj), zo, np.conj(zo)]) if j == 0 else \
            np.array([zo, np.conj(zo), zj, np.conj(zj)])
        return zeta, a_o

    def g(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        zeta, a_o = zeta_at(u, v)
        return np.where(a_o >= 0, h_of(zeta), np.nan)

    zj0 = np.sqrt(amp0[j]) * np.exp(1j * th0[j])
    start = np.array([zj0.real, zj0.imag])

    def one_turn(points: np.ndarray) -> np.ndarray:
        # map one turn, then shift along the free phase back to the chart
        zeta, _ = zeta_at(points[:, 0], points[:, 1])
        image = np.array(normal_form.N.substitute(*zeta), dtype=complex)
        tau = (th0[o] - np.angle(image[2 * o])) / w[o]
        zj = image[2 * j] * np.exp(1j * tau * w[j])
        return np.stack([zj.real, zj.imag], axis=1)

    def contour(half_width: float, closed: bool) -> tuple[list[np.ndarray] | None, float]:
        # curves of the orbit on a grid of the given half-width; None if the
        # orbit is not resolved (no contour, none through the start, too few
        # grid cells, or, if required, a curve cut by the edge of the grid)
        axis = np.linspace(-half_width, half_width, grid_size)
        step = axis[1] - axis[0]
        U, V = np.meshgrid(axis, axis, indexing="xy")
        lines = contourpy.contour_generator(x=U, y=V, z=np.ma.masked_invalid(g(U, V))).lines(h0)
        lines = [ln for ln in lines if len(ln) >= 2]
        if not lines:
            return None, step
        curves = _orbit_curves(lines, start, one_turn)
        extent = max(np.ptp(c, axis=0).max() for c in curves)
        near = _distance_to_polylines(start[None], curves)[0] <= 2 * step
        resolved = near and extent >= _MIN_CELLS * step
        if closed:
            resolved = resolved and all(np.allclose(c[0], c[-1]) for c in curves)
        return (curves if resolved else None), step

    curves, step = contour(half, closed=False)
    if curves is not None:
        return j, w, curves, g, step
    # A reduced orbit too small for the grid, typically near A_j = 0 where the
    # other plane holds almost all the energy. The orbit through the start
    # need not be small compared with the start amplitude (it can circle a
    # fixed point away from the origin), so the grid is shrunk step by step
    # and the largest grid on which the closed orbit is resolved is used.
    r0 = float(np.hypot(*start))
    if r0 <= 1e-12 * half:
        # at the origin, a fixed point of the reduced motion
        return j, w, [start[None, :]], g, step
    width = half / _ZOOM
    if width < _ZOOM_FLOOR * r0:
        return None
    while width >= _ZOOM_FLOOR * r0:
        curves, step = contour(width, closed=True)
        if curves is not None:
            return j, w, curves, g, step
        width /= _ZOOM
    # smaller than the resolution of the finest grid: the start point
    return j, w, [start[None, :]], g, step


def _refine_onto_level(
    points: np.ndarray, g: Any, level: float, step: float
) -> np.ndarray:
    """
    Move points onto a level curve by Newton's method along the gradient.

    Parameters
    ----------
    points : numpy.ndarray, shape (M, 2)
        Points near the level curve (e.g. from a contour on a grid).
    g : callable
        ``g(u, v)`` of arrays of the same shape.
    level : float
        Level.
    step : float
        Grid spacing: steps longer than it are rejected (the point is kept).

    Returns
    -------
    numpy.ndarray, shape (M, 2)
        Refined points.
    """
    p = np.array(points, dtype=float)
    eps = 1e-6 * step
    with np.errstate(invalid="ignore", divide="ignore"):
        for _ in range(_REFINE_STEPS):
            u, v = p[:, 0], p[:, 1]
            res = g(u, v) - level
            gu = (g(u + eps, v) - g(u - eps, v)) / (2 * eps)
            gv = (g(u, v + eps) - g(u, v - eps)) / (2 * eps)
            norm2 = gu * gu + gv * gv
            du, dv = -res * gu / norm2, -res * gv / norm2
            ok = np.isfinite(du) & np.isfinite(dv) & (np.hypot(du, dv) < step)
            p[ok, 0] += du[ok]
            p[ok, 1] += dv[ok]
    return p
