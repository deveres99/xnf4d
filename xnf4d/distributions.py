"""
Particle distributions matched to the non-linear motion.

:func:`generate_pencil_on_collimator` is the non-linearly matched
counterpart of ``xcoll.initial_distribution.generate_pencil_on_collimator``
(``Collimator.generate_pencil`` in ``xcoll``), with the same interface as far
as possible.

The linear pencil
-----------------
In ``xcoll`` (through ``xpart.generate_2D_pencil_with_absolute_cut``) the
pencil in the plane of the collimator is the set of points of normalised
phase space that lie beyond the jaw (``x_norm >= cut``) on circles reaching at
most ``dr`` beyond it (``r <= cut + dr``), sampled uniformly. The circles are
the linear invariants: every particle starts beyond the jaw, on an orbit
that penetrates by at most ``pencil_spread``, and the distribution is the
restriction of a matched (uniform) distribution to that region.

The non-linear pencil
---------------------
Here the circles are replaced by the invariant tori of the normal form. A
point ``z`` of normalised phase space has the normal form coordinates
``zeta = Psi(z)`` and lies on the torus of actions
``(|zeta1|^2, |zeta2|^2)``. Over many turns it visits the whole torus, i.e.
the points ``z'`` with ``Psi(z') = (|zeta1| e^{i theta1}, ...,
|zeta2| e^{i theta2}, ...)`` for all phases. Its orbit penetrates beyond the
jaw by

    D(z) = max over (theta1, theta2) of d(z'),

with ``d`` the distance beyond the jaw [m]. The points of the tori are
computed by inverting ``Psi`` with Newton's method: ``Phi`` is the inverse of
``Psi`` only up to the order of the normal form, which at the jaw amplitude
can be off by more than a typical ``pencil_spread``. The pencil is the set of
points

    d(z) >= 0  and  D(z) <= pencil_spread,

sampled uniformly in normalised phase space (a linear symplectic image of
the physical one), so that the distribution is matched to the non-linear
motion. The maximum over the phase of the other plane accounts for the
coupling of the planes: the orbit in the plane of the collimator changes
with the phase of the other plane. For a linear uncoupled map (``Psi`` and
``Phi`` the identity) this is exactly the linear pencil.

The coordinates of the other plane are drawn Gaussian in normalised
coordinates, as in ``xcoll``. Since through the coupling of the planes the
pencil region shrinks as the amplitude of the other plane grows, particles
with a large amplitude in the other plane are rarer than in a Gaussian
distribution. Candidates whose torus cannot be computed (far in the tails,
where the normal form is not valid) are discarded with a warning.

The normal form must be accurate at the jaw: the map should be fitted beyond
the jaw opening (``sigma_limit``) and the tune should not be close to a
low-order resonance, for which a non-resonant normal form does not converge.

Distributions sampled in normal form coordinates
------------------------------------------------
:func:`generate_2D_uniform_circular_sector` and
:func:`generate_matched_gaussian_bunch` are the counterparts of the
``xpart`` functions of the same name. They draw the same points as ``xpart``
(a uniform circular sector, independent Gaussians), but in normal form
coordinates ``zeta``, and map them to normalised coordinates by solving
``Psi(z) = zeta`` (as for the pencil). A density that depends only on the
normal form actions ``|zeta1|^2, |zeta2|^2`` is invariant under the one-turn
map of a non-resonant normal form, so a Gaussian bunch or a full ring is
non-linearly matched. For a linear map they are the ``xpart`` distributions.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np

from .numerical_normal_form import NormalForm4D, complex_to_real, real_to_complex
from .polynom import Polynom

__all__ = [
    "generate_2D_uniform_circular_sector",
    "generate_matched_gaussian_bunch",
    "generate_pencil_on_collimator",
]

#: Number of grid points of each refinement step of the tangent search.
_GRID = 11
#: Number of refinement steps of the tangent search.
_REFINE = 4
#: Coarse grid of the phases (plane of the collimator, other plane) used to
#: find the deepest point of a torus.
_PHASE_GRID = (16, 6)
#: Points per phase and number of steps of the local refinement.
_PHASE_LOCAL, _PHASE_REFINE = 3, 7
#: Smallest determinant of the Jacobian accepted in Newton's method.
_SINGULAR = 1e-12
#: Number of candidates processed at once.
_CHUNK = 1000
#: Accuracy of the inversion of Psi for distributions sampled in normal form
#: coordinates [beam sizes].
_TOL_INVERSE = 1e-9


# --------------------------------------------------------------------------- #
# Geometry at the collimator
# --------------------------------------------------------------------------- #


class _Geometry:
    """
    Normalised phase space at the collimator, seen from one jaw.

    Normalised coordinates ``z = (x_norm, px_norm, y_norm, py_norm)`` follow
    ``xtrack`` (``W`` matrix of the twiss, beam size from the normalised
    emittances). The physical coordinate of the plane is linear in ``z``:
    ``u = u_co + g . z``. The distance beyond the jaw at position ``pos`` is
    ``d = sign (u - pos)`` [m].

    In the plane of the collimator the coordinates are rotated to
    ``(n, t)``: ``n`` along the outward normal of the jaw (in units of beam
    size), ``t`` along the jaw.

    Parameters
    ----------
    twiss : xtrack.TwissTable
        Twiss table of the line.
    name : str
        Element at which the pencil is generated.
    plane : {"x", "y"}
        Plane of the collimator.
    side : {"+", "-"}
        Jaw: ``"+"`` (left, positive side) or ``"-"`` (right, negative side).
    pos : float
        Position [m] from which particles count as beyond the jaw.
    nemitt_x, nemitt_y : float
        Normalised emittances [m].
    """

    def __init__(
        self,
        twiss: Any,
        name: str,
        plane: Literal["x", "y"],
        side: Literal["+", "-"],
        pos: float,
        nemitt_x: float,
        nemitt_y: float,
    ) -> None:
        names = list(twiss.name)
        if name not in names:
            raise ValueError(f"Element {name!r} not found in the twiss table.")
        k = names.index(name)
        W = np.asarray(twiss.W_matrix[k])
        beta0 = float(np.asarray(twiss.particle_on_co.beta0).ravel()[0])
        gamma0 = float(np.asarray(twiss.particle_on_co.gamma0).ravel()[0])
        scale = np.sqrt(np.array([nemitt_x, nemitt_x, nemitt_y, nemitt_y]) / (beta0 * gamma0))
        row = 0 if plane == "x" else 2
        self.sign = 1.0 if side == "+" else -1.0
        self.own = (0, 1) if plane == "x" else (2, 3)
        self.other = (2, 3) if plane == "x" else (0, 1)
        # u = u_co + g . z
        self.u_co = float(np.asarray(twiss[plane])[k])
        g = W[row, :4] * scale
        self.g = g
        self.g_own, self.g_other = g[list(self.own)], g[list(self.other)]
        norm = float(np.hypot(*self.g_own))
        if norm == 0:
            raise ValueError(f"The {plane} position does not depend on the {plane} plane.")
        self.n_hat = self.sign * self.g_own / norm
        self.t_hat = np.array([-self.n_hat[1], self.n_hat[0]])
        # change of d per unit of n [m / beam size]
        self.dd_dn = norm
        self.pos = float(pos)
        if self.sign * (self.u_co - self.pos) >= 0:
            raise ValueError(
                f"The closed orbit ({self.u_co:.6g} m) is beyond the jaw at "
                f"{self.pos:.6g} m (side {side!r}): the cut is on the wrong side."
            )

    def n_at_depth(self, depth: float, other: np.ndarray) -> np.ndarray:
        """
        Normal coordinate of the line at a given distance beyond the jaw.

        Parameters
        ----------
        depth : float
            Distance beyond the jaw [m].
        other : numpy.ndarray, shape (2, M)
            Normalised coordinates of the other plane.

        Returns
        -------
        numpy.ndarray, shape (M,)
            ``n`` such that ``d = depth``.
        """
        d0 = self.sign * (self.u_co + self.g_other @ other - self.pos)
        return (depth - d0) / self.dd_dn

    def depth(self, z: np.ndarray) -> np.ndarray:
        """
        Distance beyond the jaw.

        Parameters
        ----------
        z : numpy.ndarray, shape (4, *S)
            Normalised coordinates.

        Returns
        -------
        numpy.ndarray, shape S
            ``d`` [m], positive beyond the jaw.
        """
        return self.sign * (self.u_co + np.tensordot(self.g, z, axes=1) - self.pos)

    def coordinates(self, n: np.ndarray, t: np.ndarray, other: np.ndarray) -> np.ndarray:
        """
        Normalised coordinates of points given in the rotated frame.

        Parameters
        ----------
        n, t : numpy.ndarray
            Rotated coordinates in the plane (same shape ``S``).
        other : numpy.ndarray, shape (2, *S) or broadcastable
            Normalised coordinates of the other plane.

        Returns
        -------
        numpy.ndarray, shape (4, *S)
            Normalised coordinates.
        """
        z = np.empty((4,) + np.shape(n))
        z[self.own[0]] = n * self.n_hat[0] + t * self.t_hat[0]
        z[self.own[1]] = n * self.n_hat[1] + t * self.t_hat[1]
        z[self.other[0]], z[self.other[1]] = other[0], other[1]
        return z


class _PolyEvaluator:
    """
    Fast evaluation of several polynomials at many points.

    ``Map.substitute`` evaluates every term of every component separately.
    Here every monomial is computed once, as the product of a lower monomial
    and one variable, and shared by all polynomials, which are then obtained
    by one matrix product. Points are processed in blocks to bound the
    memory used.

    Parameters
    ----------
    polys : sequence of Polynom
        Polynomials to evaluate.
    """

    _BLOCK = 8192

    def __init__(self, polys: Sequence[Polynom]) -> None:
        exps = {t.exponents for p in polys for t in p.terms}
        needed, todo = set(exps) | {(0, 0, 0, 0)}, list(exps)
        while todo:
            e = todo.pop()
            if sum(e) == 0:
                continue
            parent = self._parent(e)[0]
            if parent not in needed:
                needed.add(parent)
                todo.append(parent)
        order = sorted(needed, key=lambda e: (sum(e), e))
        index = {e: i for i, e in enumerate(order)}
        self._steps = [
            (index[e], index[self._parent(e)[0]], self._parent(e)[1])
            for e in order if sum(e) > 0
        ]
        self._size = len(order)
        self._coeffs = np.zeros((len(polys), len(order)), dtype=complex)
        for k, poly in enumerate(polys):
            for t in poly.terms:
                self._coeffs[k, index[t.exponents]] += t.coeff

    @staticmethod
    def _parent(e: tuple[int, ...]) -> tuple[tuple[int, ...], int]:
        """
        Monomial divided by its first variable.

        Parameters
        ----------
        e : tuple of 4 int
            Exponents of a non-constant monomial.

        Returns
        -------
        parent : tuple of 4 int
            Exponents of the parent monomial.
        var : int
            Index of the variable removed.
        """
        var = next(i for i in range(4) if e[i] > 0)
        return tuple(v - (i == var) for i, v in enumerate(e)), var

    def __call__(self, z: np.ndarray) -> np.ndarray:
        """
        Evaluate the polynomials.

        Parameters
        ----------
        z : numpy.ndarray, shape (4, *S)
            Values of the four (complex) variables.

        Returns
        -------
        numpy.ndarray, shape (P, *S)
            Values of the ``P`` polynomials.
        """
        z = np.asarray(z, dtype=complex)
        shape = z.shape[1:]
        flat = z.reshape(4, -1)
        out = np.empty((self._coeffs.shape[0], flat.shape[1]), dtype=complex)
        mono = np.empty((self._size, min(self._BLOCK, flat.shape[1])), dtype=complex)
        for start in range(0, flat.shape[1], self._BLOCK):
            block = flat[:, start : start + self._BLOCK]
            m = mono[:, : block.shape[1]]
            m[0] = 1.0
            for i, parent, var in self._steps:
                np.multiply(m[parent], block[var], out=m[i])
            out[:, start : start + self._BLOCK] = self._coeffs @ m
        return out.reshape((-1,) + shape)


def _to_complex(z: np.ndarray) -> np.ndarray:
    """
    Complex coordinates of real normalised coordinates.

    Parameters
    ----------
    z : numpy.ndarray, shape (4, *S)
        ``(x, px, y, py)``.

    Returns
    -------
    numpy.ndarray, shape (4, *S)
        ``(z1, z1*, z2, z2*)``.
    """
    return np.array(real_to_complex(*z))


def _to_real(w: np.ndarray) -> np.ndarray:
    """
    Real normalised coordinates of complex coordinates of real points.

    Parameters
    ----------
    w : numpy.ndarray, shape (4, *S)
        ``(z1, z1*, z2, z2*)``.

    Returns
    -------
    numpy.ndarray, shape (4, *S)
        ``(x, px, y, py)``.
    """
    return np.real(np.array(complex_to_real(*w)))


class _Transforms:
    """
    Normal form transformations of a normal form, for fast evaluation.

    Parameters
    ----------
    normal_form : NormalForm4D
        Computed normal form.
    plane : {"x", "y"}
        Plane of the collimator.
    """

    def __init__(self, normal_form: NormalForm4D, plane: str) -> None:
        psi = normal_form.Psi.polys
        self.psi = _PolyEvaluator(psi)
        self.phi = _PolyEvaluator(normal_form.Phi.polys)
        # Psi and its derivatives d Psi_i / d z_j (row 4 + 4 i + j)
        self.psi_jac = _PolyEvaluator(
            list(psi) + [Polynom.diff_Polynom(p, j) for p in psi for j in range(4)]
        )
        # index of zeta of the plane of the collimator and of the other plane
        self.own, self.other = (0, 2) if plane == "x" else (2, 0)

    def amplitudes(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Normal form amplitudes of real points.

        Parameters
        ----------
        z : numpy.ndarray, shape (4, *S)
            Normalised coordinates.

        Returns
        -------
        rho_own, rho_other : numpy.ndarray, shape S
            ``|zeta|`` of the plane of the collimator and of the other plane.
        """
        zeta = self.psi(_to_complex(z))
        return np.abs(zeta[self.own]), np.abs(zeta[self.other])

    def torus(
        self,
        rho_own: np.ndarray,
        rho_other: np.ndarray,
        th_own: np.ndarray,
        th_other: np.ndarray,
    ) -> np.ndarray:
        """
        Normal form coordinates of points of invariant tori.

        Parameters
        ----------
        rho_own, rho_other : numpy.ndarray, shape (M, 1)
            Amplitudes of the plane of the collimator and of the other plane.
        th_own, th_other : numpy.ndarray, shape (M, K) or (1, K)
            Phases.

        Returns
        -------
        numpy.ndarray, shape (4, M, K)
            ``(zeta1, zeta1*, zeta2, zeta2*)``.
        """
        a, b = np.broadcast_arrays(rho_own * np.exp(1j * th_own), rho_other * np.exp(1j * th_other))
        zeta = np.empty((4,) + a.shape, dtype=complex)
        zeta[self.own], zeta[self.own + 1] = a, np.conj(a)
        zeta[self.other], zeta[self.other + 1] = b, np.conj(b)
        return zeta

    def psi_inverse(self, zeta: np.ndarray, tol: float, max_iter: int = 30) -> np.ndarray:
        """
        Normalised coordinates ``z`` with ``Psi(z) = zeta``.

        ``Phi`` is only the inverse of ``Psi`` up to the order of the normal
        form, which is not accurate enough to resolve a pencil at large
        amplitudes. The equation is solved by Newton's method from
        ``z = Phi(zeta)``.

        Parameters
        ----------
        zeta : numpy.ndarray, shape (4, *S)
            Normal form coordinates of real points.
        tol : float
            Convergence criterion on ``max |zeta - Psi(z)|`` [beam sizes].
        max_iter : int, default 30
            Maximum number of iterations.

        Returns
        -------
        numpy.ndarray, shape (4, *S)
            Normalised coordinates, ``nan`` where the iteration did not
            converge.
        """
        shape = zeta.shape[1:]
        target = zeta.reshape(4, -1)
        z = _to_real(self.phi(target))
        active = np.arange(target.shape[1])
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            for _ in range(max_iter):
                values = self.psi_jac(_to_complex(z[:, active]))
                res = target[:, active] - values[:4]
                err = np.max(np.abs(res), axis=0)
                jac = values[4:].T.reshape(-1, 4, 4)
                # singular or invalid Jacobians: the point fails
                det = np.linalg.det(np.where(np.isfinite(jac), jac, 0.0))
                singular = ~(np.abs(det) > _SINGULAR)
                jac[singular] = np.eye(4)
                step = np.linalg.solve(jac, res.T[..., None])[..., 0].T
                z[:, active] += _to_real(step)
                # stop converged points and diverging ones (left as failed)
                keep = (err > tol) & (err < 10.0) & ~singular
                failed = ~(err <= tol) & ~keep
                z[:, active[failed]] = np.nan
                active = active[keep]
                if active.size == 0:
                    break
        z[:, active] = np.nan
        return z.reshape((4,) + shape)


def _min_along_line(
    func: Any, t_center: np.ndarray, half_width: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Minimum of functions of ``t`` by repeated grid refinement.

    Parameters
    ----------
    func : callable
        ``func(t)`` with ``t`` of shape ``(M, K)`` returns values of the
        same shape (row ``i`` is the function of point ``i``).
    t_center : numpy.ndarray, shape (M,)
        Centres of the search intervals.
    half_width : float
        Half-width of the first search interval.

    Returns
    -------
    t_min : numpy.ndarray, shape (M,)
        Position of the minimum.
    f_min : numpy.ndarray, shape (M,)
        Value of the minimum.
    """
    offsets = np.linspace(-1.0, 1.0, _GRID)
    t0, width = np.asarray(t_center, dtype=float), half_width
    rows = np.arange(t0.size)
    for _ in range(_REFINE):
        t = t0[:, None] + width * offsets[None, :]
        f = func(t)
        j = np.clip(np.argmin(f, axis=1), 1, _GRID - 2)
        f_l, f_c, f_r = f[rows, j - 1], f[rows, j], f[rows, j + 1]
        step = width * (offsets[1] - offsets[0])
        # parabola through the three points around the minimum
        den = f_l - 2 * f_c + f_r
        shift = np.where(den > 0, 0.5 * step * (f_l - f_r) / np.where(den > 0, den, 1.0), 0.0)
        t0 = t[rows, j] + np.clip(shift, -step, step)
        width = step
    return t0, np.minimum(func(t0[:, None])[:, 0], f_c)


def _torus_depth(
    transforms: _Transforms,
    geometry: _Geometry,
    rho_own: np.ndarray,
    rho_other: np.ndarray,
    tol: float,
    limit: float = np.inf,
) -> np.ndarray:
    """
    Largest distance beyond the jaw on invariant tori of the normal form.

    The deepest point is located on a coarse grid of the two phases using
    ``Phi`` (fast, approximate tori) and refined by local grids of points
    lying exactly on the tori (``Psi(z) = zeta``, see
    :meth:`_Transforms.psi_inverse`).

    Parameters
    ----------
    transforms : _Transforms
        Normal form transformations.
    geometry : _Geometry
        Geometry of the jaw.
    rho_own, rho_other : numpy.ndarray, shape (M,)
        Normal form amplitudes ``|zeta|`` of the plane of the collimator and
        of the other plane.
    tol : float
        Accuracy of the points on the tori [beam sizes].
    limit : float, default inf
        The search stops for tori found to reach deeper than ``limit`` [m]
        (any point of a torus gives a lower bound of its depth).

    Returns
    -------
    numpy.ndarray, shape (M,)
        ``max d`` over the torus [m], or a value above ``limit`` (``nan``
        where the tori cannot be computed).
    """

    # coarse search on the approximate tori z = Phi(zeta)
    n1, n2 = _PHASE_GRID
    g1, g2 = (v.ravel()[None, :] for v in np.meshgrid(
        np.arange(n1) * 2 * np.pi / n1, np.arange(n2) * 2 * np.pi / n2, indexing="ij"))
    r1, r2 = rho_own[:, None], rho_other[:, None]
    d = geometry.depth(_to_real(transforms.phi(transforms.torus(r1, r2, g1, g2))))
    best = np.argmax(d, axis=1)
    t1, t2 = g1[0, best], g2[0, best]

    # exact depth at the approximate deepest point (a lower bound)
    zeta = transforms.torus(r1, r2, t1[:, None], t2[:, None])
    d_max = geometry.depth(transforms.psi_inverse(zeta, tol))[:, 0]

    # local refinement on the exact tori, for tori not yet known to be too deep
    h1, h2 = 2 * np.pi / n1, 2 * np.pi / n2
    offsets = np.linspace(-1.0, 1.0, _PHASE_LOCAL)
    o1, o2 = (v.ravel()[None, :] for v in np.meshgrid(offsets, offsets, indexing="ij"))
    active = np.flatnonzero(d_max <= limit)
    for _ in range(_PHASE_REFINE):
        if active.size == 0:
            break
        th1 = t1[active, None] + h1 * o1
        th2 = t2[active, None] + h2 * o2
        zeta = transforms.torus(r1[active], r2[active], th1, th2)
        d = geometry.depth(transforms.psi_inverse(zeta, tol))
        failed = np.isnan(d).any(axis=1)
        best = np.argmax(np.where(np.isnan(d), -np.inf, d), axis=1)
        rows = np.arange(active.size)
        t1[active], t2[active] = th1[rows, best], th2[rows, best]
        d_max[active] = np.where(failed, np.nan, np.maximum(d_max[active], d[rows, best]))
        active = active[~failed & (d_max[active] <= limit)]
        h1, h2 = h1 / (_PHASE_LOCAL - 1), h2 / (_PHASE_LOCAL - 1)
    return d_max


# --------------------------------------------------------------------------- #
# Pencil in normalised coordinates
# --------------------------------------------------------------------------- #


def _pencil_one_jaw(
    transforms: _Transforms,
    geometry: _Geometry,
    num_particles: int,
    pencil_spread: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Non-linearly matched pencil on one jaw, in normalised coordinates.

    Candidates are sampled uniformly in a box of the plane of the collimator
    (beyond the jaw, at most ``pencil_spread`` deep, and along the jaw over
    the extent of the pencil), with Gaussian coordinates in the other plane,
    and accepted if their orbit penetrates by at most ``pencil_spread``.

    Parameters
    ----------
    transforms : _Transforms
        Normal form transformations (non-resonant normal form at the
        collimator).
    geometry : _Geometry
        Geometry of the jaw.
    num_particles : int
        Number of particles.
    pencil_spread : float
        Maximum penetration of the orbits beyond the jaw [m].
    rng : numpy.random.Generator
        Random generator.

    Returns
    -------
    numpy.ndarray, shape (4, num_particles)
        Normalised coordinates.

    Raises
    ------
    RuntimeError
        If the pencil region cannot be bracketed.
    """
    width_n = pencil_spread / geometry.dd_dn
    tol = 1e-4 * width_n

    def action_line(n: np.ndarray, other: np.ndarray):
        # action of the plane along lines of constant n, as a function of t
        def action(t: np.ndarray) -> np.ndarray:
            z = geometry.coordinates(np.broadcast_to(n[:, None], t.shape), t, other[:, :, None])
            return transforms.amplitudes(z)[0] ** 2
        return action

    # Sampling box along the jaw, from the region with the other plane at
    # zero: the orbit tangent to the line d = spread (the local minimum of the
    # action nearest to the point of that line closest to the closed orbit;
    # far along the jaw the normal form is not valid) and the extent along
    # the jaw of the region below its action (enlarged below if needed).
    zero = np.zeros((2, 1))
    n_spread = geometry.n_at_depth(pencil_spread, zero)
    n_jaw = geometry.n_at_depth(0.0, zero)
    reach = abs(float(n_spread[0]))
    t_grid = np.linspace(-reach, reach, 2001)
    j_spread = action_line(n_spread, zero)(t_grid[None, :])[0]
    minima = np.flatnonzero(
        (j_spread[1:-1] <= j_spread[:-2]) & (j_spread[1:-1] <= j_spread[2:])
    ) + 1
    if minima.size == 0:
        raise RuntimeError(
            "Could not find the orbit tangent to the pencil; the normal form may "
            "not be valid at the jaw amplitude."
        )
    i0 = minima[np.argmin(np.abs(t_grid[minima]))]
    step = t_grid[1] - t_grid[0]
    t_ref, j_ref = _min_along_line(action_line(n_spread, zero), t_grid[i0:i0 + 1], step)
    half = 2.0 * np.sqrt(2.0 * reach * width_n) + 10 * step
    for _ in range(10):
        t_local = t_ref[0] + np.linspace(-half, half, 2001)
        below = action_line(n_jaw, zero)(t_local[None, :])[0] <= j_ref[0]
        i = 1000
        if not below[i]:
            raise RuntimeError(
                "The pencil region is empty; the normal form may not be valid at "
                "the jaw amplitude."
            )
        lo, hi = i, i
        while lo > 0 and below[lo - 1]:
            lo -= 1
        while hi < t_local.size - 1 and below[hi + 1]:
            hi += 1
        if lo > 0 and hi < t_local.size - 1:
            break
        half *= 2
    else:
        raise RuntimeError("Could not bracket the pencil region along the jaw.")
    t_lo, t_hi = t_local[lo - 1], t_local[hi + 1]
    margin = 0.25 * (t_hi - t_lo)
    t_lo, t_hi = t_lo - margin, t_hi + margin

    while True:
        accepted, n_acc, n_tried, n_failed = [], 0, 0, 0
        while n_acc < num_particles:
            rate = max(n_acc, 1) / max(n_tried, 1) if n_tried else 0.3
            m = int(min(max(64, 1.3 * (num_particles - n_acc) / max(rate, 0.01)), 50_000))
            other = rng.normal(size=(2, m))
            n = geometry.n_at_depth(0.0, other) + width_n * rng.uniform(0, 1, m)
            t = rng.uniform(t_lo, t_hi, m)
            z = geometry.coordinates(n, t, other)
            rho_own, rho_other = transforms.amplitudes(z)
            depth = np.concatenate([
                _torus_depth(transforms, geometry, rho_own[s:s + _CHUNK],
                             rho_other[s:s + _CHUNK], tol, limit=pencil_spread)
                for s in range(0, m, _CHUNK)
            ])
            n_failed += int(np.isnan(depth).sum())
            ok = depth <= pencil_spread
            accepted.append(np.vstack([z[:, ok], t[ok][None]]))
            n_acc += int(ok.sum())
            n_tried += m
        out = np.concatenate(accepted, axis=1)[:, :num_particles]
        # enlarge the box if the pencil reaches its ends along the jaw
        edge = 0.02 * (t_hi - t_lo)
        if np.all((out[4] > t_lo + edge) & (out[4] < t_hi - edge)):
            break
        half = t_hi - t_lo
        t_lo, t_hi = t_lo - half / 2, t_hi + half / 2
    if n_failed:
        warnings.warn(
            f"For {n_failed} of {n_tried} candidate particles the invariant torus "
            "could not be computed (the normal form transformation does not "
            "converge there, e.g. at large amplitudes of the other plane); they "
            "were discarded.",
            stacklevel=3,
        )
    return out[:4]


# --------------------------------------------------------------------------- #
# Distributions defined in normal form coordinates
# --------------------------------------------------------------------------- #


def _check_normal_form(normal_form: NormalForm4D) -> None:
    """
    Check that a normal form is computed and non-resonant.

    Parameters
    ----------
    normal_form : NormalForm4D
        Normal form.

    Raises
    ------
    ValueError
        If the normal form is not computed or not non-resonant.
    """
    if normal_form.Psi is None:
        raise ValueError("The normal form has not been computed.")
    if normal_form.res_case != "nonresonant":
        raise ValueError(
            "The distribution is defined by the normal form actions, which are "
            "invariant only for a non-resonant normal form; got "
            f"res_case={normal_form.res_case!r}."
        )


def _from_normal_form(
    normal_form: NormalForm4D, num_particles: int, sample: Any
) -> np.ndarray:
    """
    Normalised coordinates of points sampled in normal form coordinates.

    The points ``w = (x, px, y, py)`` returned by ``sample`` are taken as the
    real form of the normal form coordinates, ``zeta = (x - i px, x + i px,
    y - i py, y + i py)``, and mapped to normalised coordinates by solving
    ``Psi(z) = zeta`` (:meth:`_Transforms.psi_inverse`). Points for which
    this fails (where the normal form is not valid) are discarded with a
    warning and replaced by new samples.

    Parameters
    ----------
    normal_form : NormalForm4D
        Computed non-resonant normal form.
    num_particles : int
        Number of points.
    sample : callable
        ``sample(m)`` returns ``m`` points ``w``, shape ``(4, m)``.

    Returns
    -------
    numpy.ndarray, shape (4, num_particles)
        Normalised coordinates.
    """
    transforms = _Transforms(normal_form, "x")
    accepted, n_acc, n_tried = [], 0, 0
    while n_acc < num_particles:
        m = num_particles - n_acc if n_tried == 0 else max(64, 2 * (num_particles - n_acc))
        z = transforms.psi_inverse(_to_complex(sample(m)), _TOL_INVERSE)
        ok = np.all(np.isfinite(z), axis=0)
        accepted.append(z[:, ok])
        n_acc += int(ok.sum())
        n_tried += m
    n_failed = n_tried - n_acc
    if n_failed:
        warnings.warn(
            f"For {n_failed} of {n_tried} sampled points the normal form "
            "transformation could not be inverted (the normal form is not valid "
            "at their amplitudes); they were discarded and resampled.",
            stacklevel=3,
        )
    return np.concatenate(accepted, axis=1)[:, :num_particles]


def generate_2D_uniform_circular_sector(
    num_particles: int,
    r_range: Sequence[float] = (0, 1),
    theta_range: Sequence[float] = (0, 2 * np.pi),
    *,
    normal_form: NormalForm4D,
    plane: Literal["x", "y"] = "x",
    sigma_other: float = 1.0,
    seed: int | np.random.Generator | None = None,
) -> tuple[np.ndarray, ...]:
    """
    Points uniformly distributed over a circular sector in normal form space.

    Non-linearly matched counterpart of
    ``xpart.generate_2D_uniform_circular_sector``. The sector (e.g. a halo
    ring ``r_range``) is sampled exactly as in ``xpart``, but in the normal
    form coordinates of ``plane``: ``r`` is the normal form amplitude and
    ``theta`` the normal form angle. The other plane is Gaussian in its normal
    form coordinates, with standard deviation ``sigma_other``. The points are
    mapped to normalised coordinates with the inverse of ``Psi``.

    A full ring (``theta_range`` of length ``2 pi``) depends only on the normal
    form actions, so it is invariant under the one-turn map (the density in
    normalised coordinates differs from that in normal form coordinates by
    ``det D Psi``, which is invariant as well and equal to 1 up to the order of
    the normal form). For a linear map (``Psi`` the identity) this is exactly
    the ``xpart`` distribution, with a Gaussian other plane.

    As the normal form couples the planes, the result is four-dimensional:
    unlike ``xpart``, the coordinates of both planes are returned.

    Parameters
    ----------
    num_particles : int
        Number of points.
    r_range : pair of float, default (0, 1)
        Range of the normal form amplitude ``(r_min, r_max)`` [beam sizes].
    theta_range : pair of float, default (0, 2 pi)
        Range of the normal form angle [rad], with the convention of ``xpart``
        (``x = r cos theta``, ``px = r sin theta`` for a linear map).
    normal_form : NormalForm4D
        Computed non-resonant normal form of the one-turn map at the element
        where the distribution is generated, in normalised coordinates.
    plane : {"x", "y"}, default "x"
        Plane of the sector.
    sigma_other : float, default 1
        Standard deviation of the other plane [beam sizes]; 0 for points at
        zero amplitude in the other plane.
    seed : int or numpy.random.Generator, optional
        Seed of the random generator.

    Returns
    -------
    x_norm, px_norm, y_norm, py_norm : numpy.ndarray, shape (num_particles,)
        Normalised coordinates.
    r, theta : numpy.ndarray, shape (num_particles,)
        Normal form amplitude and angle of each point in ``plane``.

    Raises
    ------
    ValueError
        If the arguments are invalid or the normal form is not computed or not
        non-resonant.
    """
    _check_normal_form(normal_form)
    if plane not in ("x", "y"):
        raise ValueError(f"plane must be 'x' or 'y', got {plane!r}.")
    if sigma_other < 0:
        raise ValueError(f"sigma_other must be non-negative, got {sigma_other}.")
    num_particles = int(num_particles)
    rng = np.random.default_rng(seed)
    r0, r1 = r_range
    own = [0, 1] if plane == "x" else [2, 3]
    other = [2, 3] if plane == "x" else [0, 1]

    def sample(m: int) -> np.ndarray:
        r = np.sqrt(r0 * r0 + rng.uniform(0, 1, m) * (r1 * r1 - r0 * r0))
        theta = rng.uniform(theta_range[0], theta_range[1], m)
        w = np.empty((4, m))
        w[own] = r * np.cos(theta), r * np.sin(theta)
        w[other] = sigma_other * rng.normal(size=(2, m))
        return w

    z = _from_normal_form(normal_form, num_particles, sample)
    # amplitude and angle of the returned points, from their normal form coordinates
    w = _to_real(_Transforms(normal_form, "x").psi(_to_complex(z)))
    r, theta = np.hypot(w[own[0]], w[own[1]]), np.arctan2(w[own[1]], w[own[0]])
    theta = theta_range[0] + np.mod(theta - theta_range[0], 2 * np.pi)
    return z[0], z[1], z[2], z[3], r, theta


def generate_matched_gaussian_bunch(
    num_particles: int,
    nemitt_x: float,
    nemitt_y: float,
    sigma_z: float,
    *,
    line: Any,
    normal_form: NormalForm4D,
    at_element: str,
    total_intensity_particles: float | None = None,
    engine: str | None = None,
    return_matcher: bool = False,
    twiss: Any = None,
    seed: int | np.random.Generator | None = None,
    **kwargs: Any,
) -> Any:
    """
    Generate a non-linearly matched Gaussian bunch.

    Non-linearly matched counterpart of ``xpart.generate_matched_gaussian_bunch``.
    The transverse coordinates are independent standard Gaussians, as in
    ``xpart``, but in the normal form coordinates, mapped to normalised
    coordinates with the inverse of ``Psi``. The distribution depends only on
    the normal form actions, so it is invariant under the one-turn map (see
    :func:`generate_2D_uniform_circular_sector`). For a linear map (``Psi``
    the identity) this is exactly the ``xpart`` bunch. The longitudinal
    coordinates are matched to the bucket with
    ``xpart.generate_longitudinal_coordinates``, as in ``xpart``.

    The normal form must be computed at ``at_element`` with the same
    emittances (see :func:`generate_pencil_on_collimator`), and be valid up
    to the amplitudes of the Gaussian tails: points where the transformation
    cannot be inverted are resampled with a warning.

    Parameters
    ----------
    num_particles : int
        Number of macroparticles.
    nemitt_x, nemitt_y : float
        Normalised emittances [m].
    sigma_z : float
        RMS bunch length [m].
    line : xtrack.Line
        Line with a built tracker.
    normal_form : NormalForm4D
        Computed non-resonant normal form of the one-turn map at
        ``at_element``, in normalised coordinates with emittances
        ``nemitt_x, nemitt_y``.
    at_element : str
        Element at which the bunch is generated.
    total_intensity_particles : float, optional
        Bunch intensity; the weights are 1 if not given.
    engine : str, optional
        Longitudinal matching engine, passed to
        ``xpart.generate_longitudinal_coordinates``.
    return_matcher : bool, default False
        Also return the longitudinal matcher.
    twiss : xtrack.TwissTable, optional
        Twiss table of the line (6D), computed if not given.
    seed : int or numpy.random.Generator, optional
        Seed of the random generator of the transverse coordinates (the
        longitudinal ones use the global ``numpy`` generator, as in ``xpart``).
    **kwargs
        Passed to ``line.build_particles``.

    Returns
    -------
    particles : xtrack.Particles
        The bunch at ``at_element``.
    matcher : object
        Longitudinal matcher, only if ``return_matcher`` is True.

    Raises
    ------
    ValueError
        If the tracker is not built or the normal form is not computed or not
        non-resonant.
    """
    import xpart as xp

    _check_normal_form(normal_form)
    if not line._has_valid_tracker():
        raise ValueError("Please build the tracker of the line first.")
    num_particles = int(num_particles)
    if twiss is None:
        twiss = line.twiss()
    rng = np.random.default_rng(seed)
    zeta, delta, matcher = xp.generate_longitudinal_coordinates(
        distribution="gaussian", num_particles=num_particles, line=line,
        sigma_z=sigma_z, engine=engine, return_matcher=True,
    )
    z = _from_normal_form(normal_form, num_particles, lambda m: rng.normal(size=(4, m)))
    if total_intensity_particles is None:
        total_intensity_particles = num_particles
    part = _build_particles(line, twiss, at_element, z, zeta, delta, nemitt_x, nemitt_y,
                            weight=total_intensity_particles / num_particles, **kwargs)
    return (part, matcher) if return_matcher else part


def _build_particles(
    line: Any,
    twiss: Any,
    name: str,
    z: np.ndarray,
    zeta: Any,
    delta: Any,
    nemitt_x: float,
    nemitt_y: float,
    **kwargs: Any,
) -> Any:
    """
    Particles at an element from normalised coordinates.

    Parameters
    ----------
    line : xtrack.Line
        Line.
    twiss : xtrack.TwissTable
        Twiss table of the line.
    name : str
        Element at which the particles are built.
    z : numpy.ndarray, shape (4, N)
        Normalised coordinates.
    zeta, delta : float or numpy.ndarray
        Longitudinal coordinates.
    nemitt_x, nemitt_y : float
        Normalised emittances [m].
    **kwargs
        Passed to ``line.build_particles``.

    Returns
    -------
    xtrack.Particles
        The particles.
    """
    k = list(twiss.name).index(name)
    part = line.build_particles(
        x_norm=z[0], px_norm=z[1], y_norm=z[2], py_norm=z[3],
        zeta=zeta, delta=delta, nemitt_x=nemitt_x, nemitt_y=nemitt_y,
        at_element=name, W_matrix=twiss.W_matrix[k],
        particle_on_co=twiss.get_twiss_init(at_element=name).particle_on_co,
        **kwargs,
    )
    if hasattr(part, "_init_random_number_generator"):
        part._init_random_number_generator()
    return part


# --------------------------------------------------------------------------- #
# Longitudinal coordinates
# --------------------------------------------------------------------------- #


def _longitudinal_coordinates(
    line: Any, num_particles: int, sigma_z: float, longitudinal: Any
) -> tuple[Any, Any]:
    """
    Longitudinal coordinates, as in ``xcoll``.

    Parameters
    ----------
    line : xtrack.Line
        Line.
    num_particles : int
        Number of particles.
    sigma_z : float
        Bunch length [m] for ``longitudinal="bucket"``.
    longitudinal : None, "bucket", dict or sequence
        ``None``: ``zeta = delta = 0``; ``"bucket"``: Gaussian bunch matched
        to the bucket; ``{"zeta": ..., "delta": ...}`` or ``(zeta, delta)``:
        given coordinates.

    Returns
    -------
    zeta, delta : float or numpy.ndarray
        Longitudinal coordinates.

    Raises
    ------
    ValueError
        If ``longitudinal`` is invalid.
    NotImplementedError
        For ``longitudinal="matched_dispersion"`` (as in ``xcoll``).
    """
    if longitudinal is None:
        return 0.0, 0.0
    if isinstance(longitudinal, str):
        if longitudinal == "bucket":
            import xpart as xp

            return xp.generate_longitudinal_coordinates(
                num_particles=num_particles, distribution="gaussian",
                sigma_z=sigma_z, line=line,
            )
        if longitudinal == "matched_dispersion":
            raise NotImplementedError("longitudinal='matched_dispersion' is not implemented.")
        raise ValueError(f"Unknown longitudinal option {longitudinal!r}.")
    if isinstance(longitudinal, dict):
        return longitudinal["zeta"], longitudinal["delta"]
    if hasattr(longitudinal, "__len__") and len(longitudinal) == 2:
        return longitudinal[0], longitudinal[1]
    raise ValueError(
        "longitudinal must be None, 'bucket', a dict with 'zeta' and 'delta' "
        "or a pair (zeta, delta)."
    )


# --------------------------------------------------------------------------- #
# Public interface
# --------------------------------------------------------------------------- #


def generate_pencil_on_collimator(
    line: Any,
    name: str,
    num_particles: int,
    *,
    normal_form: NormalForm4D,
    plane: Literal["x", "y"],
    jaw: float | Sequence[float],
    nemitt_x: float | None = None,
    nemitt_y: float | None = None,
    side: Literal["+-", "+", "-"] = "+-",
    pencil_spread: float = 1e-6,
    impact_parameter: float = 0.0,
    sigma_z: float = 7.61e-2,
    twiss: Any = None,
    longitudinal: Any = None,
    longitudinal_betatron_cut: Any = None,
    seed: int | np.random.Generator | None = None,
    _capacity: int | None = None,
    **kwargs: Any,
) -> Any:
    """
    Generate a non-linearly matched pencil beam on a collimator.

    Counterpart of ``xcoll.initial_distribution.generate_pencil_on_collimator``
    in which the pencil follows the invariant curves of the normal form
    instead of the linear ellipses (see the module documentation). The
    arguments shared with ``xcoll`` have the same meaning. Differences:

    - the normal form at the collimator, the plane of the collimator and the
      jaw positions are given explicitly (``xcoll`` reads the last two from
      the collimator), so ``line`` need not contain ``xcoll`` collimators;
    - the pencil is always generated at the entrance of the element
      ``name`` (``xcoll`` uses the exit of diverging jaws);
    - ``seed`` sets the random generator.

    The normal form must be computed at the element ``name`` with the same
    emittances, e.g. from a :class:`~xnf4d.one_turn_map.OneTurnMap4D` of a
    copy of the line cycled to start at ``name``, before collimators are
    installed (installed but inactive collimators do not change the map)::

        cycled = line.copy()
        cycled.cycle(name_first_element=name)
        otm = OneTurnMap4D(cycled, 8, nemitt_x, nemitt_y, sigma_limit=...)
        otm.compute()
        nf = otm.calculate_normal_form(7)

    The pencil lies at amplitudes around the jaw, so the map and normal form
    must be accurate there (``sigma_limit`` above the jaw opening).

    Parameters
    ----------
    line : xtrack.Line
        Line with a built tracker.
    name : str
        Element (collimator) at which to generate the pencil.
    num_particles : int
        Number of particles.
    normal_form : NormalForm4D
        Computed non-resonant normal form of the one-turn map at ``name``,
        in normalised coordinates with emittances ``nemitt_x, nemitt_y``.
    plane : {"x", "y"}
        Plane of the collimator (``"x"`` for a horizontal collimator, angle
        0 deg in ``xcoll``; ``"y"`` for a vertical one, 90 deg).
    jaw : float or pair of float
        Jaw positions [m]: ``(jaw_L, jaw_R)`` (left/positive and right/negative
        jaw, as ``coll.jaw_L, coll.jaw_R`` in ``xcoll``), or a single value
        ``a`` for ``(a, -a)``. Use ``None`` for a missing jaw.
    nemitt_x, nemitt_y : float, optional
        Normalised emittances [m]; by default those of the collimator
        ``line[name]`` (``xcoll`` collimators with assigned optics).
    side : {"+-", "+", "-"}, default "+-"
        Jaw(s) on which to generate the pencil: left (``"+"``), right
        (``"-"``) or both, half of the particles on each.
    pencil_spread : float, default 1e-6
        Maximum penetration of the orbits beyond the impact position [m].
    impact_parameter : float, default 0
        Distance beyond the jaw [m] from which the pencil starts.
    sigma_z : float, default 7.61e-2
        Bunch length [m] for ``longitudinal="bucket"``.
    twiss : xtrack.TwissTable, optional
        Twiss table of the line (6D), computed if not given.
    longitudinal : None, "bucket", dict or pair, optional
        Longitudinal coordinates as in ``xcoll``: ``None`` (zero),
        ``"bucket"`` (Gaussian bunch in the bucket), ``{"zeta": ...,
        "delta": ...}`` or ``(zeta, delta)``. The pencil is defined for the
        betatron coordinates (``delta = 0``), as in ``xpart``.
    longitudinal_betatron_cut : optional
        Accepted for compatibility with ``xcoll``, not used (as in
        ``xcoll``).
    seed : int or numpy.random.Generator, optional
        Seed of the random generator.
    _capacity : int, optional
        Capacity of the particles object.
    **kwargs
        Passed to ``line.build_particles``.

    Returns
    -------
    xtrack.Particles
        The particles at the element ``name``, those of the left jaw first.

    Raises
    ------
    ValueError
        If the arguments are invalid, the normal form is not computed or not
        non-resonant, or a jaw is on the wrong side of the closed orbit.
    """
    del longitudinal_betatron_cut  # unused, as in xcoll
    if not line._has_valid_tracker():
        raise ValueError("Please build tracker before generating pencil distribution!")
    _check_normal_form(normal_form)
    if plane not in ("x", "y"):
        raise ValueError(f"plane must be 'x' or 'y', got {plane!r}.")
    if side not in ("+-", "+", "-"):
        raise ValueError(f"side must be '+-', '+' or '-', got {side!r}.")
    if pencil_spread <= 0:
        raise ValueError(f"pencil_spread must be positive, got {pencil_spread}.")
    num_particles = int(num_particles)
    if num_particles < 1:
        raise ValueError(f"num_particles must be positive, got {num_particles}.")
    if nemitt_x is None or nemitt_y is None:
        element = line[name]
        if getattr(element, "optics", None) is None:
            raise ValueError(
                "nemitt_x and nemitt_y must be given (the element has no assigned optics)."
            )
        nemitt_x = element.nemitt_x if nemitt_x is None else nemitt_x
        nemitt_y = element.nemitt_y if nemitt_y is None else nemitt_y
    jaw_L, jaw_R = (jaw, -jaw) if np.ndim(jaw) == 0 else tuple(jaw)
    if twiss is None:
        twiss = line.twiss()

    sides = ("+", "-") if side == "+-" else (side,)
    counts = (
        [num_particles // 2, num_particles - num_particles // 2]
        if side == "+-" else [num_particles]
    )
    rng = np.random.default_rng(seed)
    zeta, delta = _longitudinal_coordinates(line, num_particles, sigma_z, longitudinal)

    transforms = _Transforms(normal_form, plane)
    parts = []
    for s, count in zip(sides, counts):
        if count == 0:
            continue
        jaw_pos = jaw_L if s == "+" else jaw_R
        if jaw_pos is None:
            raise ValueError(f"No jaw position given for side {s!r}.")
        pos = jaw_pos + impact_parameter if s == "+" else jaw_pos - impact_parameter
        geometry = _Geometry(twiss, name, plane, s, pos, nemitt_x, nemitt_y)
        parts.append(_pencil_one_jaw(transforms, geometry, count, pencil_spread, rng))
    z = np.concatenate(parts, axis=1)
    return _build_particles(line, twiss, name, z, zeta, delta, nemitt_x, nemitt_y,
                            _capacity=_capacity, **kwargs)
