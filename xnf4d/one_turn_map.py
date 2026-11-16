"""
One-turn maps of ``xtrack`` lines as symplectic polynomial maps.

The line is treated as a black box: particles are tracked for one turn and a
polynomial map is fitted to the result. The map acts on the transverse
normalised coordinates ``(x, px, y, py)`` (in units of the beam size, see
:meth:`OneTurnMap4D.normalised_coordinates`) and has the Lie form

    M = R o exp(:f_3:) o exp(:f_4:) o ... o exp(:f_{n+1}:),

where ``R`` is the linear one-turn map from twiss and ``f_k`` are homogeneous
polynomials (generators) of degree ``k``. Truncated at order ``n``, such a map
is symplectic up to order ``n`` by construction, whatever the generators.
The generators are found by least squares.

The fit uses two observations:

- The map coefficients enter the residuals linearly through the matrix of
  monomials of the initial coordinates. After a QR decomposition of that
  matrix the least-squares problem has ``4 x (number of monomials)``
  residuals, independently of the number of tracked particles.
- On the space of polynomials truncated at order ``n``, each ``exp(:f_k:)``
  is the exponential of a nilpotent matrix that is linear in the generator
  coefficients. The map and its exact Jacobian with respect to the
  generator coefficients follow from a few matrix products.

Convention for the Lie operators: ``:f: g = {f, g}`` with
``{f, g} = sum_(q, p) (df/dq dg/dp - df/dp dg/dq)`` over the pairs
``(x, px)`` and ``(y, py)``, and ``exp(:f:) g = sum_k :f:^k g / k!``.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from scipy import sparse
from scipy.optimize import OptimizeResult, least_squares

from .numerical_normal_form import NormalForm4D, ResonanceCase
from .polynom import Map, Polynom, Term

__all__ = ["OneTurnMap4D"]

#: Canonical pairs (q, p) of the variables (x, px, y, py).
_PAIRS = ((0, 1), (2, 3))


# --------------------------------------------------------------------------- #
# Monomials
# --------------------------------------------------------------------------- #


def _monomials(min_order: int, max_order: int) -> np.ndarray:
    """
    Exponents of all monomials in 4 variables with order in a given range.

    Monomials are sorted by total order, then lexicographically with the
    first variable varying slowest.

    Parameters
    ----------
    min_order : int
        Minimum total order.
    max_order : int
        Maximum total order.

    Returns
    -------
    numpy.ndarray of int, shape (M, 4)
        Exponents, one monomial per row.
    """
    exps = [
        (n1, n2, n3, order - n1 - n2 - n3)
        for order in range(min_order, max_order + 1)
        for n1 in range(order, -1, -1)
        for n2 in range(order - n1, -1, -1)
        for n3 in range(order - n1 - n2, -1, -1)
    ]
    return np.array(exps, dtype=np.int64).reshape(-1, 4)


def _monomial_values(coords: np.ndarray, exps: np.ndarray) -> np.ndarray:
    """
    Values of monomials at a set of points.

    Parameters
    ----------
    coords : numpy.ndarray, shape (4, N)
        Coordinates of the points.
    exps : numpy.ndarray of int, shape (M, 4)
        Exponents of the monomials.

    Returns
    -------
    numpy.ndarray, shape (N, M)
        ``out[i, m] = prod_j coords[j, i] ** exps[m, j]``.
    """
    max_exp = int(exps.max(initial=0))
    out = np.ones((coords.shape[1], len(exps)), dtype=coords.dtype)
    for j in range(4):
        powers = coords[j][:, None] ** np.arange(max_exp + 1)[None, :]
        out *= powers[:, exps[:, j]]
    return out


class _LieMapModel:
    """
    Truncated Lie map ``R o exp(:f_3:) o ... o exp(:f_{n+1}:)`` as matrices.

    Map components are stored as coefficient matrices ``C`` of shape
    ``(n_mono, 4)``: column ``i`` holds the coefficients of component ``i``
    on the monomials of orders 1 to ``n`` (see :attr:`exps`).

    Parameters
    ----------
    linear_map : numpy.ndarray, shape (4, 4)
        Linear part ``R`` of the map.
    max_order : int
        Truncation order ``n`` of the map. Generators have degrees 3 to
        ``n + 1``.
    """

    def __init__(self, linear_map: np.ndarray, max_order: int) -> None:
        if max_order < 2:
            raise ValueError(f"max_map_order must be at least 2, got {max_order}.")
        self.max_order = max_order
        #: Exponents of the map monomials (orders 1 to n).
        self.exps = _monomials(1, max_order)
        self.n_mono = len(self.exps)
        #: Degrees of the generators.
        self.degrees = list(range(3, max_order + 2))
        #: Exponents of the generator monomials, per degree.
        self.gen_exps = {k: _monomials(k, k) for k in self.degrees}
        sizes = [len(self.gen_exps[k]) for k in self.degrees]
        #: Slices of the parameter vector belonging to each degree.
        self.slices = {}
        start = 0
        for k, size in zip(self.degrees, sizes):
            self.slices[k] = slice(start, start + size)
            start += size
        #: Total number of generator coefficients.
        self.n_params = start

        base = max_order + 2
        self._strides = np.array([base**3, base**2, base, 1], dtype=np.int64)
        self._lookup = np.full(base**4, -1, dtype=np.int64)
        self._lookup[self.exps @ self._strides] = np.arange(self.n_mono)

        # Linear map: component i = sum_j R[i, j] z_j
        self.C0 = np.zeros((self.n_mono, 4))
        for j in range(4):
            unit = np.zeros(4, dtype=np.int64)
            unit[j] = 1
            self.C0[self._lookup[unit @ self._strides], :] = np.asarray(linear_map)[:, j]

        # Lie operators of the generator monomials
        self._F_pattern: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._F_values: dict[int, sparse.csr_matrix] = {}
        self._D_stack: dict[int, sparse.csr_matrix] = {}
        for k in self.degrees:
            self._build_operators(k)

    def _build_operators(self, k: int) -> None:
        """
        Sparse representations of ``:m_a:`` for the generator monomials ``m_a``.

        ``{z^a, z^e} = sum_(q, p) (a_q e_p - a_p e_q) z^(a + e - u_q - u_p)``,
        truncated at order ``n``.

        Parameters
        ----------
        k : int
            Degree of the generator monomials.
        """
        A = self.gen_exps[k]
        E = self.exps
        n = self.n_mono
        rows, cols, gens, vals = [], [], [], []
        for q, p in _PAIRS:
            coeff = A[:, None, q] * E[None, :, p] - A[:, None, p] * E[None, :, q]
            target = A[:, None, :] + E[None, :, :]
            target[..., q] -= 1
            target[..., p] -= 1
            order = target.sum(axis=2)
            ok = (coeff != 0) & (order <= self.max_order)
            ga, ge = np.nonzero(ok)
            rows.append(self._lookup[target[ga, ge] @ self._strides])
            cols.append(ge)
            gens.append(ga)
            vals.append(coeff[ga, ge].astype(float))
        rows, cols = np.concatenate(rows), np.concatenate(cols)
        gens, vals = np.concatenate(gens), np.concatenate(vals)
        n_gen = len(A)
        # Non-zero pattern of :f_k: for any f_k, and the linear map from
        # the generator coefficients to the values on that pattern
        pos, inverse = np.unique(rows * n + cols, return_inverse=True)
        self._F_pattern[k] = (pos // n, pos % n)
        self._F_values[k] = sparse.csr_matrix(
            (vals, (inverse, gens)), shape=(len(pos), n_gen)
        )
        # D_stack @ W stacks D_a @ W for all generator monomials a
        self._D_stack[k] = sparse.csr_matrix(
            (vals, (gens * n + rows, cols)), shape=(n_gen * n, n)
        )

    def lie_operator(self, k: int, theta_k: np.ndarray) -> sparse.csr_matrix:
        """
        Matrix of ``:f_k:`` on the truncated polynomial space.

        Parameters
        ----------
        k : int
            Degree of the generator.
        theta_k : numpy.ndarray
            Coefficients of the generator on the monomials ``gen_exps[k]``.

        Returns
        -------
        scipy.sparse.csr_matrix, shape (n_mono, n_mono)
            ``F`` such that the coefficients of ``{f_k, g}`` are ``F @ c_g``.
            It maps monomials of order ``j`` to order ``j + k - 2`` and is
            therefore sparse and nilpotent.
        """
        rows, cols = self._F_pattern[k]
        values = self._F_values[k] @ theta_k
        return sparse.csr_matrix((values, (rows, cols)), shape=(self.n_mono, self.n_mono))

    def _exp_terms(self, F: sparse.csr_matrix, B: np.ndarray) -> list[np.ndarray]:
        """
        Powers ``F^m B`` until they vanish.

        Parameters
        ----------
        F : scipy.sparse.csr_matrix, shape (n_mono, n_mono)
            Nilpotent Lie operator.
        B : numpy.ndarray
            Matrix to multiply.

        Returns
        -------
        list of numpy.ndarray
            ``[B, F B, F^2 B, ...]`` up to the last non-zero power.
        """
        out = [B]
        for _ in range(self.max_order):
            nxt = F @ out[-1]
            if not np.any(nxt):
                break
            out.append(nxt)
        return out

    def coefficients(self, theta: np.ndarray) -> np.ndarray:
        """
        Coefficients of the map for given generator coefficients.

        Parameters
        ----------
        theta : numpy.ndarray, shape (n_params,)
            Generator coefficients.

        Returns
        -------
        numpy.ndarray, shape (n_mono, 4)
            Map coefficients ``C``.
        """
        C = self.C0
        for k in self.degrees:
            F = self.lie_operator(k, theta[self.slices[k]])
            terms = self._exp_terms(F, C)
            C = sum(t / _factorial(m) for m, t in enumerate(terms))
        return C

    def coefficients_and_jacobian(
        self, theta: np.ndarray, left: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Map coefficients and their derivatives with respect to ``theta``.

        Parameters
        ----------
        theta : numpy.ndarray, shape (n_params,)
            Generator coefficients.
        left : numpy.ndarray, shape (m, n_mono), optional
            Matrix applied to the derivatives from the left (e.g. the
            weights of a least-squares problem). Identity by default.

        Returns
        -------
        C : numpy.ndarray, shape (n_mono, 4)
            Map coefficients.
        dC : numpy.ndarray, shape (m, 4, n_params)
            ``dC[:, i, a] = left @ dC_i / dtheta_a``.
        """
        n = self.n_mono
        # Forward pass: B_k = E_{k-1} ... E_3 C0 and the operators F_k
        Fs, Bs = {}, {}
        C = self.C0
        for k in self.degrees:
            F = self.lie_operator(k, theta[self.slices[k]])
            Fs[k], Bs[k] = F, C
            C = sum(t / _factorial(m) for m, t in enumerate(self._exp_terms(F, C)))
        # Backward pass: L_k = left @ E_K ... E_{k+1}
        L = np.eye(n) if left is None else np.array(left, dtype=float)
        m_rows = L.shape[0]
        dC = np.empty((m_rows, 4, self.n_params))
        for k in reversed(self.degrees):
            F, B = Fs[k], Bs[k]
            n_gen = len(self.gen_exps[k])
            # dE/dtheta_a B = sum_{m >= 1} 1/m! sum_{j < m} F^j D_a F^(m-1-j) B
            #               = sum_j F^j D_a W_j,  W_j = sum_i F^i B / (i + j + 1)!
            # evaluated with a Horner scheme in F. F raises the order by
            # k - 2, so F^j vanishes on the result for j > (n - 1) / (k - 2).
            Y = self._exp_terms(F, B)
            j_max = (self.max_order - 1) // (k - 2)
            X = None
            for j in range(j_max, -1, -1):
                W = sum(y / _factorial(i + j + 1) for i, y in enumerate(Y))
                DW = (self._D_stack[k] @ W).reshape(n_gen, n, 4)
                DW = DW.transpose(1, 0, 2).reshape(n, n_gen * 4)
                X = DW if X is None else F @ X + DW
            dC[:, :, self.slices[k]] = (L @ X).reshape(m_rows, n_gen, 4).transpose(0, 2, 1)
            # L <- L @ exp(F), with L @ F^m computed as (F^T)^m L^T
            terms = self._exp_terms(F.T.tocsr(), L.T)
            L = sum(t / _factorial(m) for m, t in enumerate(terms)).T
        return C, dC

    def to_map(self, C: np.ndarray) -> Map:
        """
        Polynomial map from coefficient matrices.

        Parameters
        ----------
        C : numpy.ndarray, shape (n_mono, 4)
            Map coefficients.

        Returns
        -------
        Map
            The map, without zero coefficients.
        """
        exps = self.exps.tolist()
        polys = []
        for i in range(4):
            col = C[:, i]
            polys.append(
                Polynom([Term(float(col[m]), *exps[m]) for m in np.flatnonzero(col)])
            )
        return Map(*polys)

    def generators(self, theta: np.ndarray) -> list[Polynom]:
        """
        Generators as polynomials.

        Parameters
        ----------
        theta : numpy.ndarray, shape (n_params,)
            Generator coefficients.

        Returns
        -------
        list of Polynom
            ``[f_3, f_4, ..., f_{n+1}]``.
        """
        out = []
        for k in self.degrees:
            th = theta[self.slices[k]]
            exps = self.gen_exps[k].tolist()
            out.append(Polynom([Term(float(th[a]), *exps[a]) for a in np.flatnonzero(th)]))
        return out


def _factorial(m: int) -> float:
    """
    Factorial as a float.

    Parameters
    ----------
    m : int
        Non-negative integer.

    Returns
    -------
    float
        ``m!``.
    """
    out = 1.0
    for i in range(2, m + 1):
        out *= i
    return out


def _rotation(mux: float, muy: float) -> np.ndarray:
    """
    Linear map rotating each plane by a phase advance.

    Parameters
    ----------
    mux, muy : float
        Phase advances [rad].

    Returns
    -------
    numpy.ndarray, shape (4, 4)
        Block-diagonal rotation matrix.
    """
    R = np.zeros((4, 4))
    for i, mu in ((0, mux), (2, muy)):
        c, s = np.cos(mu), np.sin(mu)
        R[i : i + 2, i : i + 2] = [[c, s], [-s, c]]
    return R


def _symplectic_rotation(R: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """
    Exact rotation closest to a linear one-turn map in normalised coordinates.

    In normalised coordinates the linear one-turn map is a rotation in each
    plane. The matrix from twiss is computed by finite differences and is
    symplectic only up to ~1e-10; the exact rotation keeps the eigenvalues
    on the unit circle.

    Parameters
    ----------
    R : numpy.ndarray, shape (4, 4)
        Linear one-turn map in normalised coordinates.
    tol : float, default 1e-6
        A warning is issued if ``R`` differs from the rotation by more.

    Returns
    -------
    numpy.ndarray, shape (4, 4)
        The rotation.
    """
    mu = [
        np.arctan2(R[i, i + 1] - R[i + 1, i], R[i, i] + R[i + 1, i + 1]) for i in (0, 2)
    ]
    out = _rotation(*mu)
    deviation = float(np.max(np.abs(R - out)))
    if deviation > tol:
        warnings.warn(
            "The linear one-turn map in normalised coordinates differs from a "
            f"rotation by {deviation:.1e}; the exact rotation is used.",
            stacklevel=3,
        )
    return out


# --------------------------------------------------------------------------- #
# One-turn map
# --------------------------------------------------------------------------- #


class OneTurnMap4D:
    """
    Symplectic polynomial one-turn map of an ``xtrack`` line.

    Particles are generated in transverse normalised coordinates, uniformly
    in the actions ``Jx, Jy`` up to ``sigma_limit**2 / 2`` and in the phases,
    tracked for one turn, and a Lie map (see the module documentation) of
    order ``max_map_order`` is fitted to the result by least squares. The
    line only needs to support tracking.

    Normalised coordinates are those of ``xtrack``
    (``line.build_particles(x_norm=...)``,
    ``TwissTable.get_normalized_coordinates``): deviations from the closed
    orbit, normalised with the ``W`` matrix of the twiss and divided by the
    beam size ``sqrt(nemitt / (beta0 gamma0))``.

    Parameters
    ----------
    line : xtrack.Line
        Line to analyse. It is only tracked, never modified.
    max_map_order : int
        Order of the polynomial map (at least 2).
    nemitt_x : float
        Horizontal normalised emittance [m], defines the unit of the
        normalised coordinates.
    nemitt_y : float
        Vertical normalised emittance [m].
    sigma_limit : float, default 10
        Maximum amplitude of the fitted region, in units of the beam size.
    num_particles : int, optional
        Number of particles tracked for the fit. By default 10 times the
        number of monomials of the map (at least 1000). Lost particles are
        excluded with a warning.
    method : {"6d", "4d"}, default "6d"
        Twiss method. With ``"6d"`` the particles start on the longitudinal
        closed orbit and the transverse part of the 6D one-turn map is
        fitted. With ``"4d"`` the longitudinal coordinates are frozen while
        tracking.
    seed : int or numpy.random.Generator, optional
        Seed of the random generator used to sample the particles.
    twiss_kwargs : dict, optional
        Additional arguments passed to ``line.twiss``.

    Notes
    -----
    The fit is global: the error is spread over the whole region up to
    ``sigma_limit``. Where the map is strongly non-linear (large detuning,
    resonance islands), a low order lets the large residuals at the edge of
    the region spoil the accuracy inside it. Increasing ``max_map_order``
    (orders 8 to 10 take seconds to a minute) or reducing ``sigma_limit``
    to the region of interest improves the map there.

    :meth:`from_data` fits a map to tracking data obtained in any other way.

    Attributes
    ----------
    one_turn_map : Map or None
        Fitted map in real normalised coordinates ``(x, px, y, py)``.
    generators : list of Polynom or None
        Fitted generators ``[f_3, ..., f_{n+1}]``.
    fit_result : scipy.optimize.OptimizeResult or None
        Result of the least-squares fit.
    fit_rms : numpy.ndarray or None
        RMS residual of the fit for each coordinate, in normalised units.
    normal_form : NormalForm4D or None
        Normal form computed by :meth:`calculate_normal_form`.
    R : numpy.ndarray, shape (4, 4)
        Linear one-turn map in normalised coordinates: the rotation by the
        phase advances of the twiss, kept fixed in the fit.
    z0, z1 : numpy.ndarray, shape (4, N)
        Normalised coordinates of the surviving particles before and after
        one turn.
    """

    def __init__(
        self,
        line: Any,
        max_map_order: int,
        nemitt_x: float,
        nemitt_y: float,
        sigma_limit: float = 10.0,
        num_particles: int | None = None,
        method: Literal["6d", "4d"] = "6d",
        seed: int | np.random.Generator | None = None,
        twiss_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if method not in ("6d", "4d"):
            raise ValueError(f"method must be '6d' or '4d', got {method!r}.")
        if sigma_limit <= 0:
            raise ValueError(f"sigma_limit must be positive, got {sigma_limit}.")
        self.line = line
        self.max_map_order = int(max_map_order)
        self.nemitt_x = float(nemitt_x)
        self.nemitt_y = float(nemitt_y)
        self.sigma_limit = float(sigma_limit)
        self.method = method

        if nemitt_x <= 0 or nemitt_y <= 0:
            raise ValueError("The emittances nemitt_x and nemitt_y must be positive.")

        self.twiss = line.twiss(method=method, **(twiss_kwargs or {}))
        W = self.twiss.W_matrix[0]
        R = np.linalg.inv(W) @ self.twiss.R_matrix @ W
        self._setup(_symplectic_rotation(R[:4, :4]), max_map_order)

        if num_particles is None:
            num_particles = max(1000, 10 * self._model.n_mono)
        rng = np.random.default_rng(seed)
        z0 = self._sample(rng, int(num_particles))
        self.z0, self.z1 = self._track(z0)

    def _setup(self, linear_map: np.ndarray, max_map_order: int) -> None:
        """
        Initialise the fit model and the result attributes.

        Parameters
        ----------
        linear_map : numpy.ndarray, shape (4, 4)
            Linear one-turn map in normalised coordinates.
        max_map_order : int
            Order of the polynomial map.

        Raises
        ------
        ValueError
            If ``max_map_order`` is below 2.
        """
        self.max_map_order = int(max_map_order)
        if self.max_map_order < 2:
            raise ValueError(f"max_map_order must be at least 2, got {max_map_order}.")
        self.R: np.ndarray = np.array(linear_map, dtype=float)
        self._model = _LieMapModel(self.R, self.max_map_order)
        self.one_turn_map: Map | None = None
        self.generators: list[Polynom] | None = None
        self.fit_result: OptimizeResult | None = None
        self.fit_rms: np.ndarray | None = None
        self.normal_form: NormalForm4D | None = None

    @classmethod
    def from_data(
        cls,
        z0: np.ndarray,
        z1: np.ndarray,
        linear_map: np.ndarray,
        max_map_order: int,
        sigma_limit: float | None = None,
    ) -> OneTurnMap4D:
        """
        One-turn map fitted to given tracking data.

        No line is attached: the methods that build or track particles are
        not available.

        Parameters
        ----------
        z0 : array_like, shape (4, N)
            Initial normalised coordinates.
        z1 : array_like, shape (4, N)
            Normalised coordinates after one turn.
        linear_map : array_like, shape (4, 4)
            Linear one-turn map, kept fixed in the fit.
        max_map_order : int
            Order of the polynomial map.
        sigma_limit : float, optional
            Amplitude scale used to condition the fit. By default the largest
            amplitude of ``z0``.

        Returns
        -------
        OneTurnMap4D
            The (not yet fitted) map; call :meth:`compute`.

        Raises
        ------
        ValueError
            If the data have the wrong shape or are too few.
        """
        z0 = np.asarray(z0, dtype=float)
        z1 = np.asarray(z1, dtype=float)
        if z0.ndim != 2 or z0.shape[0] != 4 or z0.shape != z1.shape:
            raise ValueError(
                f"z0 and z1 must both have shape (4, N), got {z0.shape} and {z1.shape}."
            )
        obj = object.__new__(cls)
        obj.line = None
        obj.twiss = None
        obj.method = None
        obj.nemitt_x = obj.nemitt_y = None
        obj._setup(linear_map, max_map_order)
        if z0.shape[1] < 2 * obj._model.n_mono:
            raise ValueError(
                f"{z0.shape[1]} samples are too few to fit "
                f"{obj._model.n_mono} monomials per coordinate."
            )
        if sigma_limit is None:
            sigma_limit = float(np.max(np.abs(z0)))
        obj.sigma_limit = float(sigma_limit)
        obj.z0, obj.z1 = z0, z1
        return obj

    # ------------------------------------------------------------------ #
    # Normalised coordinates and tracking
    # ------------------------------------------------------------------ #

    def _require_line(self) -> None:
        """
        Check that a line is attached.

        Raises
        ------
        RuntimeError
            If the map was created with :meth:`from_data`.
        """
        if self.line is None:
            raise RuntimeError("No line is attached to this map (created with from_data).")

    def _sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """
        Random initial conditions, uniform in actions and phases.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random generator.
        n : int
            Number of points.

        Returns
        -------
        numpy.ndarray, shape (4, n)
            Normalised coordinates.
        """
        J = rng.uniform(0, self.sigma_limit**2 / 2, (2, n))
        phi = rng.uniform(0, 2 * np.pi, (2, n))
        r = np.sqrt(2 * J)
        return np.array(
            [r[0] * np.cos(phi[0]), r[0] * np.sin(phi[0]),
             r[1] * np.cos(phi[1]), r[1] * np.sin(phi[1])]
        )

    def build_particles(self, z: np.ndarray) -> Any:
        """
        Particles at given normalised coordinates.

        The longitudinal coordinates are set to the closed orbit.

        Parameters
        ----------
        z : array_like, shape (4, N)
            Normalised coordinates ``(x, px, y, py)``.

        Returns
        -------
        xtrack.Particles
            The particles.
        """
        self._require_line()
        z = np.atleast_2d(np.asarray(z, dtype=float))
        return self.line.build_particles(
            method=self.method,
            x_norm=z[0], px_norm=z[1], y_norm=z[2], py_norm=z[3],
            zeta_norm=np.zeros(z.shape[1]), pzeta_norm=np.zeros(z.shape[1]),
            nemitt_x=self.nemitt_x, nemitt_y=self.nemitt_y,
            W_matrix=self.twiss.W_matrix[0],
            particle_on_co=self.twiss.particle_on_co,
        )

    def normalised_coordinates(self, particles: Any) -> np.ndarray:
        """
        Normalised coordinates of particles, sorted by particle id.

        Parameters
        ----------
        particles : xtrack.Particles
            Particles at the start of the line.

        Returns
        -------
        numpy.ndarray, shape (4, N)
            Normalised coordinates ``(x, px, y, py)``.
        """
        self._require_line()
        tab = self.twiss.get_normalized_coordinates(
            particles, nemitt_x=self.nemitt_x, nemitt_y=self.nemitt_y
        )
        order = np.argsort(np.asarray(tab.particle_id))
        return np.array(
            [np.asarray(getattr(tab, c))[order] for c in ("x_norm", "px_norm", "y_norm", "py_norm")]
        )

    def track(self, z: np.ndarray, num_turns: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """
        Track particles through the line.

        Parameters
        ----------
        z : array_like, shape (4, N)
            Initial normalised coordinates.
        num_turns : int, default 1
            Number of turns.

        Returns
        -------
        z_out : numpy.ndarray, shape (4, N)
            Final normalised coordinates (``nan`` for lost particles).
        alive : numpy.ndarray of bool, shape (N,)
            Which particles survived.
        """
        particles = self.build_particles(z)
        kwargs = {"freeze_longitudinal": True} if self.method == "4d" else {}
        self.line.track(particles, num_turns=num_turns, **kwargs)
        order = np.argsort(np.asarray(particles.particle_id))
        alive = np.asarray(particles.state)[order] > 0
        z_out = self.normalised_coordinates(particles)
        z_out[:, ~alive] = np.nan
        return z_out, alive

    def _track(self, z0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Track the fit sample for one turn, dropping lost particles.

        Parameters
        ----------
        z0 : numpy.ndarray, shape (4, N)
            Requested initial normalised coordinates.

        Returns
        -------
        z0, z1 : numpy.ndarray, shape (4, N_alive)
            Initial and final normalised coordinates of surviving particles.

        Raises
        ------
        ValueError
            If too few particles survive for the fit.
        """
        z0 = self.normalised_coordinates(self.build_particles(z0))
        z1, alive = self.track(z0)
        n_lost = int(np.sum(~alive))
        if n_lost:
            warnings.warn(
                f"{n_lost} of {alive.size} particles were lost during the turn "
                "and are excluded from the fit; consider reducing sigma_limit.",
                stacklevel=3,
            )
        if np.sum(alive) < 2 * self._model.n_mono:
            raise ValueError(
                f"Only {int(np.sum(alive))} particles survived, too few to fit "
                f"{self._model.n_mono} monomials per coordinate."
            )
        return z0[:, alive], z1[:, alive]

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #

    def _least_squares_system(self) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Reduced least-squares system equivalent to the fit on all particles.

        With ``V`` the matrix of map monomials at the initial coordinates and
        ``V = Q R_V``, ``sum ||V C - z1||^2 = ||R_V C - Q^T z1||^2 + rest``.

        Returns
        -------
        weights : numpy.ndarray, shape (n_mono, n_mono)
            ``R_V``.
        target : numpy.ndarray, shape (n_mono, 4)
            ``Q^T z1``.
        rest : float
            Part of the sum of squares that no polynomial of this order can
            fit.
        """
        exps = self._model.exps
        scale = self.sigma_limit
        # Monomials of the scaled coordinates are well conditioned
        V = _monomial_values(self.z0 / scale, exps)
        Q, RV = np.linalg.qr(V, mode="reduced")
        weights = RV * scale ** exps.sum(axis=1).astype(float)[None, :]
        target = Q.T @ self.z1.T
        rest = float(np.sum(self.z1**2) - np.sum(target**2))
        return weights, target, max(rest, 0.0)

    def compute(self, verbose: int = 0, **least_squares_kwargs: Any) -> None:
        """
        Fit the one-turn map.

        Sets :attr:`one_turn_map`, :attr:`generators`, :attr:`fit_result`
        and :attr:`fit_rms`.

        Parameters
        ----------
        verbose : int, default 0
            Verbosity of :func:`scipy.optimize.least_squares`.
        **least_squares_kwargs
            Further arguments for :func:`scipy.optimize.least_squares`
            (defaults: ``method="lm"``, ``x_scale="jac"``).
        """
        model = self._model
        weights, target, _ = self._least_squares_system()
        cache: dict[str, Any] = {}

        def evaluate(theta: np.ndarray) -> None:
            if cache.get("theta") is not None and np.array_equal(cache["theta"], theta):
                return
            C, dC = model.coefficients_and_jacobian(theta, left=weights)
            cache.update(
                theta=theta.copy(),
                res=(weights @ C - target).ravel(),
                jac=dC.reshape(-1, model.n_params),
            )

        def fun(theta: np.ndarray) -> np.ndarray:
            evaluate(theta)
            return cache["res"]

        def jac(theta: np.ndarray) -> np.ndarray:
            evaluate(theta)
            return cache["jac"]

        options = {"method": "lm", "x_scale": "jac"}
        options.update(least_squares_kwargs)
        result = least_squares(
            fun, np.zeros(model.n_params), jac=jac, verbose=verbose, **options
        )
        if not result.success:
            warnings.warn(f"The fit did not converge: {result.message}", stacklevel=2)
        theta = result.x
        self.fit_result = result
        self.one_turn_map = model.to_map(model.coefficients(theta))
        self.generators = model.generators(theta)
        pred = np.array(self.one_turn_map.substitute(*self.z0)).real
        self.fit_rms = np.sqrt(np.mean((pred - self.z1) ** 2, axis=1))

    def track_map(self, z: np.ndarray, num_turns: int = 1) -> np.ndarray:
        """
        Iterate the fitted one-turn map.

        Parameters
        ----------
        z : array_like, shape (4, N)
            Initial normalised coordinates.
        num_turns : int, default 1
            Number of turns.

        Returns
        -------
        numpy.ndarray, shape (4, N)
            Normalised coordinates after ``num_turns`` turns.
        """
        if self.one_turn_map is None:
            self.compute()
        z = np.array(z, dtype=float)
        for _ in range(num_turns):
            z = np.array(self.one_turn_map.substitute(*z)).real
        return z

    # ------------------------------------------------------------------ #
    # Complex map and normal form
    # ------------------------------------------------------------------ #

    @property
    def complex_one_turn_map(self) -> Map:
        """
        The one-turn map in complex normalised coordinates.

        With ``z1 = x - i px`` and ``z2 = y - i py``, the map acts on
        ``(z1, z1*, z2, z2*)``. Computed (and fitted) on first access.

        Returns
        -------
        Map
            The complex map, truncated at :attr:`max_map_order`.
        """
        if self.one_turn_map is None:
            self.compute()
        n = self.max_map_order

        def lin(*terms: tuple[complex, tuple[int, int, int, int]]) -> Polynom:
            return Polynom([Term(c, *e) for c, e in terms])

        to_real = Map(
            lin((0.5, (1, 0, 0, 0)), (0.5, (0, 1, 0, 0))),
            lin((0.5j, (1, 0, 0, 0)), (-0.5j, (0, 1, 0, 0))),
            lin((0.5, (0, 0, 1, 0)), (0.5, (0, 0, 0, 1))),
            lin((0.5j, (0, 0, 1, 0)), (-0.5j, (0, 0, 0, 1))),
        )
        to_complex = Map(
            lin((1.0, (1, 0, 0, 0)), (-1j, (0, 1, 0, 0))),
            lin((1.0, (1, 0, 0, 0)), (1j, (0, 1, 0, 0))),
            lin((1.0, (0, 0, 1, 0)), (-1j, (0, 0, 0, 1))),
            lin((1.0, (0, 0, 1, 0)), (1j, (0, 0, 0, 1))),
        )
        inner = Map.composition_Map(to_real, self.one_turn_map, n)
        return Map.composition_Map(inner, to_complex, n)

    def calculate_normal_form(
        self,
        max_nf_order: int,
        res_case: ResonanceCase = "nonresonant",
        res_eig: Sequence[complex] | None = None,
        res_basis1: Sequence[int] | None = None,
        res_basis2: Sequence[int] | None = None,
    ) -> NormalForm4D:
        """
        Compute the normal form of the one-turn map.

        Parameters
        ----------
        max_nf_order : int
            Maximum order of the normal form.
        res_case : {"nonresonant", "resonant", "quasiresonant"}, default "nonresonant"
            Type of normal form (see :class:`NormalForm4D`).
        res_eig : sequence of 4 complex, optional
            Resonant eigenvalues (quasi-resonant case).
        res_basis1 : sequence of 2 int, optional
            First resonance vector ``[n, m]``.
        res_basis2 : sequence of 2 int, optional
            Second resonance vector.

        Returns
        -------
        NormalForm4D
            The computed normal form, also stored in :attr:`normal_form`.

        Warns
        -----
        UserWarning
            If ``max_map_order < max_nf_order + 1``. A normal form of order
            ``n`` uses the terms of the map up to order ``n + 1``; the
            missing orders of a fitted map are then treated as zero, which
            spoils the normal form at large amplitudes.
        """
        if self.max_map_order < max_nf_order + 1:
            warnings.warn(
                f"A normal form of order {max_nf_order} uses the map up to order "
                f"{max_nf_order + 1}, but the map is of order {self.max_map_order}: "
                "the missing terms are treated as zero. Use max_map_order >= "
                f"{max_nf_order + 1} or max_nf_order <= {self.max_map_order - 1}.",
                stacklevel=2,
            )
        nf = NormalForm4D(
            self.complex_one_turn_map,
            max_nf_order,
            res_case,
            res_eig=res_eig,
            res_basis1=res_basis1,
            res_basis2=res_basis2,
        )
        nf.compute_normal_form()
        self.normal_form = nf
        return nf
