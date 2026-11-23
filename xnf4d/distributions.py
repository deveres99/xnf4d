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
:func:`generate_matched_qgaussian_bunch` does the same with q-Gaussians,
factorisable (independent planes) or not (a function of the 4D radius),
optionally cut in normalised coordinates.

Joining normal forms
--------------------
A single polynomial map, hence a single normal form, may not be accurate
over the whole amplitude range of a distribution (e.g. with beam-beam, whose
kick is far from polynomial). The bunch functions therefore accept several
normal forms (e.g. from maps fitted on different amplitude ranges, see
``OneTurnMap4D(..., sigma_min=...)``) joined at seams of constant
``|zeta1|^2 + |zeta2|^2``: a level set of an invariant of all of them (if
non-resonant or with the coupling resonance), hence a set of tori of each.
Each part of the distribution then depends on the invariants of its normal
form; the remaining error, a thin layer at each seam where the two normal
forms disagree, is measured by :func:`seam_mismatch`.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import gamma

from .numerical_normal_form import NormalForm4D, complex_to_real, real_to_complex
from .polynom import Polynom

__all__ = [
    "generate_2D_uniform_circular_sector",
    "generate_matched_gaussian_bunch",
    "generate_matched_qgaussian_bunch",
    "generate_pencil_on_collimator",
    "SeamMismatch",
    "seam_mismatch",
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
#: Points sampled in normal form coordinates beyond this multiple of a cut
#: are discarded without inverting Psi.
_CUT_MARGIN = 2.0
#: Largest number of points sampled at once in normal form coordinates.
_MAX_BATCH = 200_000
#: Points sampled per requested point before a cut is deemed too tight.
_MAX_TRIES = 1000
#: Largest number of seams joining normal forms in one distribution.
_MAX_SEAMS = 3


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
# Distributions sampled in normal form coordinates
# --------------------------------------------------------------------------- #


def _from_normal_form(
    normal_form: NormalForm4D | Sequence[NormalForm4D],
    num_particles: int,
    sample: Callable[[int], np.ndarray],
    cut: Callable[[np.ndarray, float], np.ndarray] | None = None,
    seams: Sequence[float] | None = None,
) -> np.ndarray:
    """
    Normalised coordinates of points sampled in normal form coordinates.

    The points ``w = (x, px, y, py)`` returned by ``sample`` are taken as the
    real form of the normal form coordinates, ``zeta = (x - i px, x + i px,
    y - i py, y + i py)``, and mapped to normalised coordinates by solving
    ``Psi(z) = zeta`` (:meth:`_Transforms.psi_inverse`). Points for which
    this fails (where the normal form is not valid) are discarded with a
    warning and replaced by new samples.

    With a ``cut``, only points within the cut in normalised coordinates are
    kept, and new samples are drawn until there are ``num_particles`` of
    them. Points beyond the cut enlarged by ``_CUT_MARGIN`` in normal form
    coordinates are discarded without inverting ``Psi``: as long as the
    normal form is valid, ``Psi`` changes the amplitudes much less than that.
    Failed inversions are reported only for points within the cut in normal
    form coordinates (the others would most likely be cut).

    With several normal forms joined by ``seams``, a point with normal form
    amplitude ``rho = sqrt(|zeta1|^2 + |zeta2|^2)`` between ``seams[k - 1]``
    and ``seams[k]`` is mapped with the inverse of ``Psi`` of
    ``normal_form[k]`` (see :func:`seam_mismatch`).

    Parameters
    ----------
    normal_form : NormalForm4D or sequence of NormalForm4D
        Computed normal form(s), from the innermost.
    num_particles : int
        Number of points.
    sample : callable
        ``sample(m)`` returns ``m`` points ``w``, shape ``(4, m)``.
    cut : callable, optional
        ``cut(u, scale)`` returns whether the points ``u``, shape ``(4, m)``,
        lie within the cut with its limits multiplied by ``scale``.
    seams : sequence of float, optional
        Increasing seam radii [beam sizes], one fewer than the normal forms.

    Returns
    -------
    numpy.ndarray, shape (4, num_particles)
        Normalised coordinates.

    Raises
    ------
    RuntimeError
        If almost no sampled point lies within the cut.
    """
    forms = [normal_form] if isinstance(normal_form, NormalForm4D) or not isinstance(
        normal_form, Sequence) else list(normal_form)
    transforms = [_Transforms(nf, "x") for nf in forms]
    seams = np.asarray([] if seams is None else seams, dtype=float)
    accepted, n_acc, n_tried, n_failed = [], 0, 0, 0
    while n_acc < num_particles:
        missing = num_particles - n_acc
        rate = max(n_acc, 1) / n_tried if n_tried else 1.0
        m = int(min(max(64, 1.2 * missing / rate), _MAX_BATCH)) if n_tried else missing
        w = sample(m)
        if cut is None:
            candidate = inside = np.ones(m, dtype=bool)
        else:
            candidate, inside = cut(w, _CUT_MARGIN), cut(w, 1.0)
        z = np.full((4, m), np.nan)
        # normal form of each point, from its normal form amplitude
        region = np.searchsorted(seams, np.sqrt(np.sum(w**2, axis=0)), side="right")
        for k, transform in enumerate(transforms):
            sel = candidate & (region == k)
            if sel.any():
                z[:, sel] = transform.psi_inverse(_to_complex(w[:, sel]), _TOL_INVERSE)
        finite = np.all(np.isfinite(z), axis=0)
        n_failed += int(np.sum(candidate & inside & ~finite))
        ok = finite if cut is None else finite & cut(np.where(finite, z, 0.0), 1.0)
        accepted.append(z[:, ok])
        n_acc += int(ok.sum())
        n_tried += m
        if n_acc < num_particles and n_tried > _MAX_TRIES * num_particles + 1_000_000:
            raise RuntimeError(
                f"Only {n_acc} of {n_tried} sampled points lie within the cut; "
                "the cut is too tight for the distribution."
            )
    if n_failed:
        warnings.warn(
            f"For {n_failed} of {n_tried} sampled points the normal form "
            "transformation could not be inverted (the normal form is not valid "
            "at their amplitudes); they were discarded and resampled.",
            stacklevel=3,
        )
    return np.concatenate(accepted, axis=1)[:, :num_particles]


def _normalised_cut(
    sigma_cut_x: float | None, sigma_cut_y: float | None, skew_cut: bool
) -> Callable[[np.ndarray, float], np.ndarray] | None:
    """
    Cut on the radii of the two planes in normalised phase space.

    With ``r_x = hypot(x, px)`` and ``r_y = hypot(y, py)``, the points kept
    are those with ``r_x^2 / sigma_cut_x^2 + r_y^2 / sigma_cut_y^2 < 1``
    (``skew_cut``, as by a skew collimator) or ``r_x < sigma_cut_x`` and
    ``r_y < sigma_cut_y``. A missing limit is infinite (no cut in that plane).

    Parameters
    ----------
    sigma_cut_x, sigma_cut_y : float or None
        Limits [beam sizes], positive.
    skew_cut : bool
        Elliptical cut in ``(r_x, r_y)`` if True, rectangular otherwise.

    Returns
    -------
    callable or None
        ``cut(u, scale)``, whether the points ``u``, shape ``(4, m)``, lie
        within the cut with the limits multiplied by ``scale``; None if both
        limits are missing.

    Raises
    ------
    ValueError
        If a limit is not positive.
    """
    for name, value in (("sigma_cut_x", sigma_cut_x), ("sigma_cut_y", sigma_cut_y)):
        if value is not None and not value > 0:
            raise ValueError(f"{name} must be positive, got {value}.")
    if sigma_cut_x is None and sigma_cut_y is None:
        return None
    lim_x = np.inf if sigma_cut_x is None else float(sigma_cut_x)
    lim_y = np.inf if sigma_cut_y is None else float(sigma_cut_y)

    def cut(u: np.ndarray, scale: float) -> np.ndarray:
        a = np.hypot(u[0], u[1]) / (scale * lim_x)
        b = np.hypot(u[2], u[3]) / (scale * lim_y)
        return a * a + b * b < 1 if skew_cut else (a < 1) & (b < 1)

    return cut


# --------------------------------------------------------------------------- #
# q-Gaussian helpers
# --------------------------------------------------------------------------- #


def _f_2D(r: ArrayLike, q: float, beta: float) -> np.ndarray:
    """
    Density of the 2D q-Gaussian as a function of the radius.

    The radially symmetric density in the plane ``(u, pu)``

        f(r) = C_2 (1 + beta (q - 1) r^2)^(-(q + 1) / (2 (q - 1))),
        C_2 = beta (3 - q) / (2 pi),

    with ``r^2 = u^2 + pu^2``, normalised to 1 over the plane. The exponent is
    chosen such that the 1D marginal of each coordinate is the 1D q-Gaussian
    ``e_q(-beta u^2) ~ (1 + beta (q - 1) u^2)^(-1 / (q - 1))``, of variance
    ``1 / (beta (5 - 3 q))`` (finite for ``q < 5/3``). It is a bivariate
    Student t distribution with ``(3 - q) / (q - 1)`` degrees of freedom.

    Parameters
    ----------
    r : array_like
        Radius.
    q : float
        Parameter ``q``, ``1 < q < 3`` (the formula cannot evaluate the
        Gaussian limit ``q = 1``).
    beta : float
        Scale parameter, positive.

    Returns
    -------
    numpy.ndarray
        Density at ``r`` (per unit area).
    """

    C_2 = (beta * (3 - q)) / (2 * np.pi)

    return C_2 * (1 + beta * (q - 1) * r**2)**((q + 1) / (2 - 2*q))


def _f_4D(m: ArrayLike, q: float, beta: float) -> np.ndarray:
    """
    Density of the 4D q-Gaussian as a function of the 4D radius.

    The spherically symmetric density in ``(x, px, y, py)``

        f(m) = C_4 (1 + beta (q - 1) m^2)^(-(3 q - 1) / (2 (q - 1))),
        C_4 = beta^2 (3 - q) (q + 1) / (4 pi^2),

    with ``m^2 = x^2 + px^2 + y^2 + py^2``, normalised to 1 over the 4D space.
    It does not factorise into a product of densities of the two planes. The
    exponent is chosen such that the 1D marginal of each coordinate is the
    same 1D q-Gaussian as for :func:`_f_2D` (and the 2D marginal of each plane
    is :func:`_f_2D`). It is a 4D Student t distribution with
    ``(3 - q) / (q - 1)`` degrees of freedom.

    Parameters
    ----------
    m : array_like
        4D radius.
    q : float
        Parameter ``q``, ``1 < q < 3``.
    beta : float
        Scale parameter, positive.

    Returns
    -------
    numpy.ndarray
        Density at ``m`` (per unit 4D volume).
    """

    C_4 = (beta**2 * (3 - q) * (q + 1)) / (4 * np.pi**2)

    return C_4 * (1 + beta * (q - 1) * m**2)**((3*q - 1) / (2 - 2*q))


def _check_q_parameters(q: float, beta: float) -> None:
    """
    Check the parameters of the q-Gaussians.

    Parameters
    ----------
    q : float
        Parameter ``q``, ``1 <= q < 3``.
    beta : float
        Scale parameter, positive.

    Raises
    ------
    ValueError
        If ``q`` or ``beta`` is out of range.
    """
    if not 1 <= q < 3:
        raise ValueError(f"q must satisfy 1 <= q < 3, got {q}.")
    if not beta > 0:
        raise ValueError(f"beta must be positive, got {beta}.")


def _q_gaussian_vectors(
    num_particles: int, dim: int, q: float, beta: float, rng: np.random.Generator
) -> np.ndarray:
    """
    Sample the radially symmetric q-Gaussian in 2 or 4 dimensions.

    The densities :func:`_f_2D` (``dim = 2``) and :func:`_f_4D` (``dim = 4``)
    are multivariate Student t distributions with ``nu = (3 - q) / (q - 1)``
    degrees of freedom and scale ``s = 1 / sqrt((3 - q) beta)``, so they are
    sampled exactly (without truncation) as

        w = s z / sqrt(g / nu),

    with ``z`` a standard Gaussian vector and ``g`` an independent chi-squared
    variable with ``nu`` degrees of freedom. For ``q = 1`` (``nu`` infinite)
    this is the Gaussian of variance ``1 / (2 beta)``, the limit of the
    q-Gaussian.

    Parameters
    ----------
    num_particles : int
        Number of points.
    dim : {2, 4}
        Dimension.
    q : float
        Parameter ``q``, ``1 <= q < 3``. The tails fall as ``r^(-(3 - q) /
        (q - 1))`` (probability beyond ``r``), very slowly as ``q`` approaches
        3.
    beta : float
        Scale parameter, positive.
    rng : numpy.random.Generator
        Random generator.

    Returns
    -------
    numpy.ndarray, shape (dim, num_particles)
        Coordinates.

    Raises
    ------
    ValueError
        If ``q`` or ``beta`` is out of range.
    """
    _check_q_parameters(q, beta)
    z = rng.normal(size=(dim, num_particles))
    if q == 1:
        return z * np.sqrt(1 / (2 * beta))
    nu = (3 - q) / (q - 1)
    g = rng.chisquare(nu, size=num_particles)
    return z * np.sqrt(nu / ((3 - q) * beta * g))


def _generate_nonfactorizable(
    num_particles: int, q: float, beta: float, rng: np.random.Generator
) -> np.ndarray:
    """
    Sample the non-factorisable 4D q-Gaussian.

    Points with the density :func:`_f_4D`, which depends only on the 4D
    radius ``m``: isotropic in the 4D space, but not uniform, and the two
    planes are not independent. Taken as normal form coordinates,
    ``m^2 = |zeta1|^2 + |zeta2|^2`` is a function of the normal form actions,
    so the distribution is invariant under the one-turn map of a
    non-resonant normal form, and of one with a coupling resonance
    (``(1, -1)``, which keeps ``|zeta1|^2 + |zeta2|^2`` invariant).

    Parameters
    ----------
    num_particles : int
        Number of points.
    q : float
        Parameter ``q``, ``1 <= q < 3``.
    beta : float
        Scale parameter, positive.
    rng : numpy.random.Generator
        Random generator.

    Returns
    -------
    numpy.ndarray, shape (4, num_particles)
        Coordinates ``(x, px, y, py)``.
    """
    return _q_gaussian_vectors(num_particles, 4, q, beta, rng)


def _generate_factorizable(
    num_particles: int, q: float, beta: float, rng: np.random.Generator
) -> np.ndarray:
    """
    Sample the factorisable 4D q-Gaussian.

    Points with the density ``f(r_x) f(r_y)`` of :func:`_f_2D`, the product
    of independent 2D q-Gaussians of the two planes. Taken as normal form
    coordinates, ``r_x = |zeta1|`` and ``r_y = |zeta2|``, so the distribution
    is invariant under the one-turn map of a non-resonant normal form (not
    under one with a resonance, which exchanges ``r_x`` and ``r_y``).

    Parameters
    ----------
    num_particles : int
        Number of points.
    q : float
        Parameter ``q``, ``1 <= q < 3``.
    beta : float
        Scale parameter, positive.
    rng : numpy.random.Generator
        Random generator.

    Returns
    -------
    numpy.ndarray, shape (4, num_particles)
        Coordinates ``(x, px, y, py)``.
    """
    return np.concatenate([
        _q_gaussian_vectors(num_particles, 2, q, beta, rng),
        _q_gaussian_vectors(num_particles, 2, q, beta, rng),
    ])


# --------------------------------------------------------------------------- #
# Checks, longitudinal coordinates and particles
# --------------------------------------------------------------------------- #


def _normal_form_resonance(normal_form: NormalForm4D) -> tuple[int, int] | None:
    """
    Resonance kept in a normal form.

    Parameters
    ----------
    normal_form : NormalForm4D
        Computed normal form.

    Returns
    -------
    tuple of 2 int or None
        Primitive resonance vector ``(n, m)`` (``n Qx + m Qy = p``), or None
        for a non-resonant normal form.

    Raises
    ------
    ValueError
        For a normal form with two resonances.
    """
    case = getattr(normal_form, "res_case", None)
    if case is None:  # not a NormalForm4D: read the resonance from H
        from .orbits import _resonance_of

        return _resonance_of(normal_form.H)
    if case == "nonresonant":
        return None
    if getattr(normal_form, "res_space_dim", 1) > 1:
        raise ValueError(
            "Normal forms with two resonances are not supported: only h is "
            "invariant, and no distribution of a beam can be built from it."
        )
    n, m = int(normal_form.res_basis1[0]), int(normal_form.res_basis2[0])
    g = np.gcd(abs(n), abs(m))
    n, m = n // g, m // g
    if n < 0 or (n == 0 and m < 0):
        n, m = -n, -m
    return n, m


def _check_normal_form(
    normal_form: NormalForm4D,
    density: Literal["sum", "other"] | None,
    what: str,
) -> None:
    """
    Check that a normal form is computed and suits a distribution.

    A distribution is invariant under the one-turn map in normal form if its
    density depends only on the invariants. For a non-resonant normal form
    these are the actions ``A1 = |zeta1|^2`` and ``A2 = |zeta2|^2``. With one
    resonance ``(n, m)`` they are ``h`` and ``m A1 - n A2``: a density
    depending on ``A1 + A2`` only (``density="sum"``) is invariant for the
    coupling resonances (``n + m = 0``); otherwise the distribution is only
    approximately matched, and a warning is issued.

    Parameters
    ----------
    normal_form : NormalForm4D
        Normal form.
    density : {"sum", "other"} or None
        How the density depends on the actions: through ``A1 + A2`` only, or
        otherwise; None if the distribution requires a non-resonant normal
        form.
    what : str
        Name of the distribution, for the messages.

    Raises
    ------
    ValueError
        If the normal form is not computed, has two resonances, or is
        resonant when ``density`` is None.

    Warns
    -----
    UserWarning
        If the distribution is not exactly invariant under a resonant normal
        form.
    """
    if normal_form.Psi is None:
        raise ValueError("The normal form has not been computed.")
    resonance = _normal_form_resonance(normal_form)
    if resonance is None:
        return
    n, m = resonance
    if density is None:
        raise ValueError(
            f"The {what} needs a non-resonant normal form (its invariant tori "
            f"are those of constant actions); got the resonance ({n}, {m})."
        )
    if density == "sum" and n + m == 0:
        return
    warnings.warn(
        f"With the resonance ({n}, {m}) the invariants of the normal form are h "
        f"and {m} A1 - {n} A2 (A = |zeta|^2); the density of the {what} is not a "
        "function of them, so it is only approximately matched.",
        stacklevel=3,
    )


def _normal_forms_and_seams(
    normal_form: NormalForm4D | Sequence[NormalForm4D],
    seams: Sequence[float] | None,
    density: Literal["sum", "other"],
    what: str,
) -> tuple[list[NormalForm4D], np.ndarray]:
    """
    Check one normal form, or several joined by seams.

    Parameters
    ----------
    normal_form : NormalForm4D or sequence of NormalForm4D
        Normal form, or normal forms from the innermost.
    seams : sequence of float or None
        Seam radii (one fewer than the normal forms), or None for one normal
        form.
    density : {"sum", "other"}
        How the density depends on the actions (see
        :func:`_check_normal_form`).
    what : str
        Name of the distribution, for the messages.

    Returns
    -------
    normal_forms : list of NormalForm4D
        The normal forms.
    seams : numpy.ndarray
        The seam radii (empty for one normal form).

    Raises
    ------
    ValueError
        If the normal forms or seams are invalid, or the density is not a
        function of ``|zeta1|^2 + |zeta2|^2`` with several normal forms.
    """
    if isinstance(normal_form, NormalForm4D) or not isinstance(normal_form, Sequence):
        if seams is not None and len(seams) > 0:
            raise ValueError("seams require a sequence of normal forms.")
        _check_normal_form(normal_form, density, what)
        return [normal_form], np.empty(0)
    forms = list(normal_form)
    seams = np.asarray([] if seams is None else seams, dtype=float)
    if not 1 <= len(forms) - 1 <= _MAX_SEAMS:
        raise ValueError(
            f"Between 2 and {_MAX_SEAMS + 1} normal forms can be joined, got {len(forms)}."
        )
    if seams.ndim != 1 or seams.size != len(forms) - 1:
        raise ValueError(
            f"{len(forms)} normal forms need {len(forms) - 1} seam radii, got {seams.tolist()}."
        )
    if np.any(seams <= 0) or np.any(np.diff(seams) <= 0):
        raise ValueError(f"The seam radii must be positive and increasing, got {seams.tolist()}.")
    if density != "sum":
        raise ValueError(
            f"Normal forms can only be joined for a density depending on "
            f"|zeta1|^2 + |zeta2|^2 alone, which the {what} is not."
        )
    for k, nf in enumerate(forms):
        _seam_invariant_check(nf, f"normal_form[{k}]")
    return forms, seams


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


def _matched_bunch(
    num_particles: int,
    nemitt_x: float,
    nemitt_y: float,
    sigma_z: float,
    line: Any,
    normal_form: NormalForm4D,
    at_element: str,
    sample: Callable[[int], np.ndarray],
    cut: Callable[[np.ndarray, float], np.ndarray] | None,
    total_intensity_particles: float | None,
    engine: str | None,
    return_matcher: bool,
    twiss: Any,
    density: Literal["sum", "other"],
    what: str,
    seams: Sequence[float] | None = None,
    **kwargs: Any,
) -> Any:
    """
    Bunch with transverse coordinates sampled in normal form coordinates.

    Shared by :func:`generate_matched_gaussian_bunch` and
    :func:`generate_matched_qgaussian_bunch` (see there for the arguments).
    The longitudinal coordinates are a Gaussian bunch matched to the bucket
    (``xpart.generate_longitudinal_coordinates``), the transverse ones are
    those of :func:`_from_normal_form` with ``sample`` and ``cut``.
    ``density`` and ``what`` describe the transverse density for
    :func:`_check_normal_form`; ``normal_form`` can be a sequence of normal
    forms joined by ``seams`` (see :func:`_normal_forms_and_seams`).

    Returns
    -------
    particles : xtrack.Particles
        The bunch at ``at_element``.
    matcher : object
        Longitudinal matcher, only if ``return_matcher`` is True.

    Raises
    ------
    ValueError
        If the tracker is not built, or the normal form is not computed or
        has two resonances.
    """
    import xpart as xp

    forms, seams = _normal_forms_and_seams(normal_form, seams, density, what)
    if not line._has_valid_tracker():
        raise ValueError("Please build the tracker of the line first.")
    num_particles = int(num_particles)
    if twiss is None:
        twiss = line.twiss()
    zeta, delta, matcher = xp.generate_longitudinal_coordinates(
        distribution="gaussian", num_particles=num_particles, line=line,
        sigma_z=sigma_z, engine=engine, return_matcher=True,
    )
    z = _from_normal_form(forms, num_particles, sample, cut, seams)
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
    _check_normal_form(normal_form, None, "pencil")
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
        Computed normal form of the one-turn map at the element where the
        distribution is generated, in normalised coordinates: non-resonant,
        or with one resonance (the sector is then only approximately
        matched, with a warning: see :func:`_check_normal_form`).
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
        If the arguments are invalid or the normal form is not computed or
        has two resonances.
    """
    _check_normal_form(normal_form, "other", "circular sector")
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
    normal_form: NormalForm4D | Sequence[NormalForm4D],
    at_element: str,
    seams: Sequence[float] | None = None,
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
        Computed normal form of the one-turn map at ``at_element``, in
        normalised coordinates with emittances ``nemitt_x, nemitt_y``:
        non-resonant or with one resonance. The bunch is exactly matched for
        a non-resonant normal form and for a coupling resonance (``(1, -1)``,
        which keeps ``|zeta1|^2 + |zeta2|^2`` invariant), otherwise
        approximately, with a warning (see :func:`_check_normal_form`).
        A sequence of 2 to 4 normal forms (non-resonant or with the coupling
        resonance, e.g. from maps fitted on different amplitude ranges) is
        joined at the radii ``seams``.
    at_element : str
        Element at which the bunch is generated.
    seams : sequence of float, optional
        With several normal forms, the increasing normal form amplitudes
        ``rho = sqrt(|zeta1|^2 + |zeta2|^2)`` [beam sizes] at which they are
        joined: points sampled with ``rho`` between ``seams[k - 1]`` and
        ``seams[k]`` use ``normal_form[k]``. Each seam is a set of tori of
        both neighbouring normal forms, so the bunch stays matched up to
        their mismatch there (check it with :func:`seam_mismatch`).
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
        If the tracker is not built, the normal form is not computed or has
        two resonances, or the normal forms joined by seams are invalid.
    """
    rng = np.random.default_rng(seed)
    return _matched_bunch(
        num_particles, nemitt_x, nemitt_y, sigma_z, line, normal_form, at_element,
        lambda m: rng.normal(size=(4, m)), None, total_intensity_particles, engine,
        return_matcher, twiss, "sum", "Gaussian bunch", seams=seams, **kwargs,
    )


def generate_matched_qgaussian_bunch(
    num_particles: int,
    nemitt_x: float,
    nemitt_y: float,
    sigma_z: float,
    *,
    line: Any,
    normal_form: NormalForm4D | Sequence[NormalForm4D],
    at_element: str,
    q: float,
    beta: float,
    fact: bool,
    sigma_cut_x: float | None = None,
    sigma_cut_y: float | None = None,
    skew_cut: bool = True,
    seams: Sequence[float] | None = None,
    total_intensity_particles: float | None = None,
    engine: str | None = None,
    return_matcher: bool = False,
    twiss: Any = None,
    seed: int | np.random.Generator | None = None,
    **kwargs: Any,
) -> Any:
    """
    Generate a non-linearly matched q-Gaussian bunch.

    Counterpart of :func:`generate_matched_gaussian_bunch` with q-Gaussian
    transverse coordinates. The points are sampled in normal form
    coordinates, either

    - factorisable (``fact=True``): the product of independent 2D q-Gaussians
      of the two planes, of density ``f(r_x) f(r_y)`` (:func:`_f_2D`), or
    - non-factorisable (``fact=False``): the 4D q-Gaussian of density
      ``f(m)`` (:func:`_f_4D`), depending only on the 4D radius
      ``m^2 = r_x^2 + r_y^2``, with planes that are not independent,

    with the same 1D q-Gaussian ``e_q(-beta u^2)`` as marginal of each
    coordinate. With a normal form of the coupling resonance only the
    non-factorisable model is exactly matched, as the resonance exchanges
    the actions of the planes (the factorisable one is generated with a
    warning). They are mapped to normalised coordinates with the inverse of
    ``Psi``. Both densities depend only on the normal form actions, so they
    are invariant under the one-turn map. ``q = 1`` is the Gaussian bunch of
    variance ``1 / (2 beta)`` (both models then coincide).

    The emittances only set the unit of the normalised coordinates (the
    beam size of a Gaussian beam of that emittance); the width of the
    q-Gaussian in these units is set by ``beta`` alone (variance
    ``1 / (beta (5 - 3 q))`` of each coordinate for a linear map, infinite for
    ``q >= 5/3``).

    With ``sigma_cut_x`` and/or ``sigma_cut_y``, particles are kept only
    within the cut on the radii ``r_x = hypot(x_norm, px_norm)`` and
    ``r_y = hypot(y_norm, py_norm)`` in normalised coordinates:
    ``r_x^2 / sigma_cut_x^2 + r_y^2 / sigma_cut_y^2 < 1`` (``skew_cut=True``,
    as by a skew collimator) or ``r_x < sigma_cut_x`` and ``r_y < sigma_cut_y``
    (``skew_cut=False``); a missing limit means no cut in that plane. Points
    are sampled until exactly ``num_particles`` lie within the cut. The cut is
    applied in normalised coordinates, where it is not a function of the
    normal form actions: near its edge the cut distribution is not exactly
    invariant (as a beam cut by collimators).

    The longitudinal coordinates are a Gaussian bunch matched to the bucket,
    as in :func:`generate_matched_gaussian_bunch`. The normal form must be
    computed at ``at_element`` with the same emittances and be valid up to
    the cut (without a cut, up to the amplitudes of the tails): points where
    the transformation cannot be inverted are resampled with a warning.

    Parameters
    ----------
    num_particles : int
        Number of macroparticles.
    nemitt_x, nemitt_y : float
        Normalised emittances [m], the scale of the normalised coordinates.
    sigma_z : float
        RMS bunch length [m].
    line : xtrack.Line
        Line with a built tracker.
    normal_form : NormalForm4D
        Computed normal form of the one-turn map at ``at_element``, in
        normalised coordinates with emittances ``nemitt_x, nemitt_y``:
        non-resonant or with one resonance. The bunch is exactly matched for
        a non-resonant normal form and for a coupling resonance (``(1, -1)``,
        which keeps ``|zeta1|^2 + |zeta2|^2`` invariant), otherwise
        approximately, with a warning (see :func:`_check_normal_form`).
        A sequence of 2 to 4 normal forms (non-resonant or with the coupling
        resonance, e.g. from maps fitted on different amplitude ranges) is
        joined at the radii ``seams``.
    at_element : str
        Element at which the bunch is generated.
    q : float
        Parameter ``q`` of the q-Gaussian, ``1 <= q < 3``.
    beta : float
        Scale parameter of the q-Gaussian, positive [1 / beam size^2].
    fact : bool
        Factorisable (True) or non-factorisable (False) distribution.
    sigma_cut_x, sigma_cut_y : float, optional
        Limits of the cut on ``r_x`` and ``r_y`` [beam sizes], positive. No
        cut if neither is given.
    skew_cut : bool, default True
        Elliptical cut in ``(r_x, r_y)`` (skew collimator) if True,
        rectangular cut otherwise.
    seams : sequence of float, optional
        With several normal forms (non-factorisable model only), the
        increasing normal form amplitudes at which they are joined (see
        :func:`generate_matched_gaussian_bunch`).
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
    TypeError
        If ``fact`` or ``skew_cut`` is not a bool.
    ValueError
        If ``q``, ``beta`` or a cut limit is out of range, the tracker is not
        built, the normal form is not computed or has two resonances, or the
        normal forms joined by seams are invalid (or factorisable).
    RuntimeError
        If the cut is too tight for the distribution.
    """
    _check_q_parameters(q, beta)
    for name, value in (("fact", fact), ("skew_cut", skew_cut)):
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must be a bool, got {value!r}.")
    cut = _normalised_cut(sigma_cut_x, sigma_cut_y, bool(skew_cut))
    generate = _generate_factorizable if fact else _generate_nonfactorizable
    rng = np.random.default_rng(seed)
    return _matched_bunch(
        num_particles, nemitt_x, nemitt_y, sigma_z, line, normal_form, at_element,
        lambda m: generate(m, q, beta, rng), cut, total_intensity_particles, engine,
        return_matcher, twiss, "sum" if (not fact or q == 1) else "other",
        f"{'factorisable' if fact else 'non-factorisable'} q-Gaussian bunch",
        seams=seams, **kwargs,
    )


# --------------------------------------------------------------------------- #
# Joining normal forms
# --------------------------------------------------------------------------- #


@dataclass
class SeamMismatch:
    """
    Mismatch of two normal forms on seams of constant ``|zeta1|^2 + |zeta2|^2``.

    Returned by :func:`seam_mismatch`. A distribution can be generated with
    one normal form (``inner``) below a seam radius ``rho`` and another one
    (``outer``) above, the seam being the level set ``rho^2 = |zeta1|^2 +
    |zeta2|^2`` of an invariant of both. If the two normal forms were exact,
    the seam would be the same set of tori in both. The differences
    ``delta = rho_other(z) - rho`` of the seam radius seen by the other normal
    form, at points ``z`` of the seam of one of them, measure the mismatch:
    the inner part then ends at a slightly different place than where the
    outer part begins, leaving a thin layer of thickness about ``|delta|``
    that is covered twice (``delta > 0`` on the inner seam) or not at all
    (``delta < 0``). The fraction of particles affected is about
    ``p(rho) |delta|``, with ``p`` the density of the distribution in ``rho``.

    Attributes
    ----------
    radii : numpy.ndarray, shape (R,)
        Seam radii ``rho`` [beam sizes].
    inner_to_outer, outer_to_inner : list of numpy.ndarray
        For each radius, ``delta`` at points of the seam of the inner normal
        form seen by the outer one, and vice versa [beam sizes] (points
        where ``Psi`` could not be inverted are left out).
    failed : numpy.ndarray, shape (R, 2)
        Fraction of the seam points of the inner and of the outer normal form
        at which ``Psi`` could not be inverted.
    """

    radii: np.ndarray
    inner_to_outer: list[np.ndarray]
    outer_to_inner: list[np.ndarray]
    failed: np.ndarray

    def summary(self) -> dict[str, np.ndarray]:
        """
        Statistics of the mismatch for each seam radius.

        Returns
        -------
        dict of numpy.ndarray, shape (R,)
            ``"mean"``, ``"rms"`` and ``"max"`` (of ``|delta|``) for each
            direction (keys prefixed by ``"inner_to_outer_"`` and
            ``"outer_to_inner_"``), and ``"rms"`` of both together.
        """
        out = {}
        for name, data in (("inner_to_outer", self.inner_to_outer),
                           ("outer_to_inner", self.outer_to_inner)):
            out[f"{name}_mean"] = np.array([np.mean(d) if d.size else np.nan for d in data])
            out[f"{name}_rms"] = np.array([np.sqrt(np.mean(d**2)) if d.size else np.nan for d in data])
            out[f"{name}_max"] = np.array([np.max(np.abs(d)) if d.size else np.nan for d in data])
        both = [np.concatenate([a, b]) for a, b in zip(self.inner_to_outer, self.outer_to_inner)]
        out["rms"] = np.array([np.sqrt(np.mean(d**2)) if d.size else np.nan for d in both])
        return out

    @property
    def best_radius(self) -> float:
        """
        Seam radius with the smallest RMS mismatch.

        Returns
        -------
        float
            The radius [beam sizes].
        """
        rms = self.summary()["rms"]
        return float(self.radii[np.nanargmin(rms)])


def _seam_invariant_check(normal_form: NormalForm4D, name: str) -> None:
    """
    Check that ``|zeta1|^2 + |zeta2|^2`` is invariant for a normal form.

    Parameters
    ----------
    normal_form : NormalForm4D
        Normal form.
    name : str
        Name of the argument, for the message.

    Raises
    ------
    ValueError
        If the normal form is not computed, or keeps a resonance other than
        the coupling resonances ``(n, -n)``.
    """
    if normal_form.Psi is None:
        raise ValueError(f"The normal form {name} has not been computed.")
    resonance = _normal_form_resonance(normal_form)
    if resonance is not None and resonance[0] + resonance[1] != 0:
        raise ValueError(
            f"The normal form {name} keeps the resonance {resonance}, for which "
            "|zeta1|^2 + |zeta2|^2 is not invariant: a seam on it would not be "
            "made of tori. Seams are supported for non-resonant normal forms "
            "and the coupling resonance (1, -1)."
        )


def seam_mismatch(
    inner: NormalForm4D,
    outer: NormalForm4D,
    radii: float | Sequence[float],
    num_points: int = 2000,
    seed: int | np.random.Generator | None = None,
) -> SeamMismatch:
    """
    Mismatch of two normal forms joined on seams of constant 4D amplitude.

    For each radius ``rho``, points are drawn uniformly on the seam
    ``|zeta1|^2 + |zeta2|^2 = rho^2`` of each normal form (uniform phases,
    ``|zeta1|^2`` uniform in ``[0, rho^2]``), mapped to normalised
    coordinates with the inverse of its ``Psi`` and to the normal form
    coordinates of the other one, where their radius is compared with
    ``rho`` (see :class:`SeamMismatch`). Both normal forms must keep
    ``|zeta1|^2 + |zeta2|^2`` invariant (non-resonant, or with the coupling
    resonance), and be computed on the same normalised coordinates (same
    element, emittances and twiss), e.g. from one-turn maps fitted on
    different amplitude ranges (``OneTurnMap4D(..., sigma_min=...)``).

    Parameters
    ----------
    inner, outer : NormalForm4D
        Normal forms used below and above the seam.
    radii : float or sequence of float
        Seam radii ``rho`` to test [beam sizes].
    num_points : int, default 2000
        Number of points per seam and normal form.
    seed : int or numpy.random.Generator, optional
        Seed of the random generator.

    Returns
    -------
    SeamMismatch
        The mismatch for each radius.

    Raises
    ------
    ValueError
        If a normal form is not computed or does not keep
        ``|zeta1|^2 + |zeta2|^2`` invariant, or the arguments are invalid.
    """
    _seam_invariant_check(inner, "inner")
    _seam_invariant_check(outer, "outer")
    radii = np.atleast_1d(np.asarray(radii, dtype=float))
    if radii.ndim != 1 or np.any(radii <= 0):
        raise ValueError("radii must be positive.")
    num_points = int(num_points)
    if num_points < 1:
        raise ValueError(f"num_points must be positive, got {num_points}.")
    rng = np.random.default_rng(seed)
    t_in, t_out = _Transforms(inner, "x"), _Transforms(outer, "x")
    results = {"inner_to_outer": [], "outer_to_inner": []}
    failed = np.zeros((radii.size, 2))
    for k, rho in enumerate(radii):
        for col, (name, src, dst) in enumerate(
            (("inner_to_outer", t_in, t_out), ("outer_to_inner", t_out, t_in))
        ):
            u = rng.uniform(0, 1, num_points)
            th = rng.uniform(0, 2 * np.pi, (2, num_points))
            zeta = src.torus(rho * np.sqrt(u), rho * np.sqrt(1 - u), th[0], th[1])
            z = src.psi_inverse(zeta, _TOL_INVERSE)
            ok = np.all(np.isfinite(z), axis=0)
            failed[k, col] = 1 - ok.mean()
            seen = dst.psi(_to_complex(z[:, ok]))
            rho_seen = np.sqrt(np.abs(seen[0]) ** 2 + np.abs(seen[2]) ** 2)
            results[name].append(rho_seen - rho)
    return SeamMismatch(
        radii=radii, inner_to_outer=results["inner_to_outer"],
        outer_to_inner=results["outer_to_inner"], failed=failed,
    )
