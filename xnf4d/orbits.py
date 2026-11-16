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
    "Orbits",
    "hamiltonian_orbits",
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
