"""
Numerical 4D normal forms of symplectic one-turn maps.

Implementation of the normal form algorithm of A. Bazzani, M. Giovannozzi,
G. Servizi, E. Todesco and G. Turchetti, "Resonant normal forms,
interpolating Hamiltonians and stability analysis of area preserving maps",
Physica D 64 (1993) (see also CERN SL/94-02 (AP), https://cds.cern.ch/record/262179).

Conventions
-----------
All maps act on complex coordinates. The complex Courant-Snyder coordinates
are ``(z1, z1*, z2, z2*)`` with ``z1 = x - i px`` and ``z2 = y - i py``
(see :func:`real_to_complex` and :func:`complex_to_real`). The normal form
coordinates are ``(zeta1, zeta1*, zeta2, zeta2*)``.

For a one-turn map ``F`` the algorithm computes

- ``Phi``: normal form -> Courant-Snyder coordinates, ``F o Phi = Phi o N``;
- ``Psi``: Courant-Snyder -> normal form coordinates, the inverse of ``Phi``;
- ``N``: the one-turn map in normal form, ``N = Psi o F o Phi``;
- ``U``: the interpolating vector field of ``N``, a "logarithm" of the map
  in normal form. With ``Lambda = diag(lambda_j)`` the linear part of ``F``,
  ``Lambda_res = diag(res_eig)`` and ``flow(V)`` the time-one flow of a
  vector field ``V``:

  - non-resonant case: ``N = flow(U)``;
  - resonant case: ``N = Lambda o flow(U - U_lin)``, where ``U_lin`` is the
    linear part ``log(lambda_j) zeta_j`` of ``U``;
  - quasi-resonant case: ``N = Lambda_res o flow(U)``;

- ``H``: the interpolating Hamiltonian of ``U``, i.e.
  ``U_0 = dH/dzeta1*``, ``U_1 = -dH/dzeta1``,
  ``U_2 = dH/dzeta2*``, ``U_3 = -dH/dzeta2``.

Internally, polynomials are stored as arrays of coefficients, indexed by
monomial: monomials are sorted by total order, then by the exponents
``(n1, n2, n3)`` in increasing lexicographic order.
"""

from __future__ import annotations

import gzip
import json
import os
from collections.abc import Sequence
from functools import lru_cache
from importlib import metadata
from typing import Any, Literal

import numpy as np

from .polynom import ArrayOrScalar, Map, Polynom, Term

#: Type of the ``res_case`` argument of :class:`NormalForm4D`.
ResonanceCase = Literal["nonresonant", "resonant", "quasiresonant"]
_RES_CASES = ("nonresonant", "resonant", "quasiresonant")

#: Identifier and version of the file format of :meth:`NormalForm4D.to_dict`.
_FORMAT_NAME = "xnf4d.NormalForm4D"
_FORMAT_VERSION = 1

__all__ = ["NormalForm4D", "real_to_complex", "complex_to_real"]

# Exponent shifts of the four "conjugate" partners used by the homological
# equation: component j of a map gains (k[0][j], k[1][j]) in (n1-n2, n3-n4).
_K_SHIFT = np.array([[1, -1, 0, 0], [0, 0, 1, -1]])

_ZERO_ATOL = 1e-16

# Index of the four linear monomials z1, z1*, z2, z2* in the coefficient
# arrays (orders are sorted with n1 varying slowest).
_LINEAR_IDX = (4, 3, 2, 1)


def real_to_complex(
    x: ArrayOrScalar,
    px: ArrayOrScalar,
    y: ArrayOrScalar,
    py: ArrayOrScalar,
) -> tuple[ArrayOrScalar, ArrayOrScalar, ArrayOrScalar, ArrayOrScalar]:
    """
    Convert real Courant-Snyder coordinates to complex ones.

    Parameters
    ----------
    x, px, y, py : float, complex or numpy.ndarray
        Normalised Courant-Snyder coordinates.

    Returns
    -------
    tuple of 4 complex or numpy.ndarray
        ``(z1, z1*, z2, z2*)`` with ``z1 = x - i px`` and ``z2 = y - i py``.
    """
    return (x - 1j * px, x + 1j * px, y - 1j * py, y + 1j * py)


def complex_to_real(
    z1: ArrayOrScalar,
    z1s: ArrayOrScalar,
    z2: ArrayOrScalar,
    z2s: ArrayOrScalar,
) -> tuple[ArrayOrScalar, ArrayOrScalar, ArrayOrScalar, ArrayOrScalar]:
    """
    Convert complex Courant-Snyder coordinates to real ones.

    This is the inverse of :func:`real_to_complex`. The results are returned
    as complex numbers: for physical inputs (``z1s == conj(z1)`` and
    ``z2s == conj(z2)``) their imaginary parts vanish up to rounding, and
    ``.real`` can be taken.

    Parameters
    ----------
    z1, z1s, z2, z2s : complex or numpy.ndarray
        Complex Courant-Snyder coordinates ``(z1, z1*, z2, z2*)``.

    Returns
    -------
    tuple of 4 complex or numpy.ndarray
        ``(x, px, y, py)``.
    """
    return (
        (z1 + z1s) / 2.0,
        1j * (z1 - z1s) / 2.0,
        (z2 + z2s) / 2.0,
        1j * (z2 - z2s) / 2.0,
    )


def _n_monomials(order: int) -> int:
    """
    Number of monomials in 4 variables of total order at most ``order``.

    Parameters
    ----------
    order : int
        Maximum total order (``-1`` gives ``0``).

    Returns
    -------
    int
        ``binomial(order + 4, 4)``.
    """
    return (order + 1) * (order + 2) * (order + 3) * (order + 4) // 24


def _first(order: int) -> int:
    """
    Index of the first monomial ``(0, 0, 0, order)`` of a given order.

    Parameters
    ----------
    order : int
        Total order.

    Returns
    -------
    int
        Array index of the monomial.
    """
    return _n_monomials(order - 1)


def _last(order: int) -> int:
    """
    Index of the last monomial ``(order, 0, 0, 0)`` of a given order.

    Parameters
    ----------
    order : int
        Total order.

    Returns
    -------
    int
        Array index of the monomial.
    """
    return _n_monomials(order) - 1


@lru_cache(maxsize=None)
def _monomial_exponents(max_order: int) -> np.ndarray:
    """
    Exponents of all monomials up to a given order, in array-index order.

    Parameters
    ----------
    max_order : int
        Maximum total order.

    Returns
    -------
    numpy.ndarray of int, shape (N, 4)
        Row ``i`` holds the exponents of the monomial with index ``i``.
    """
    exps = [
        (n1, n2, n3, order - n1 - n2 - n3)
        for order in range(max_order + 1)
        for n1 in range(order + 1)
        for n2 in range(order - n1 + 1)
        for n3 in range(order - n1 - n2 + 1)
    ]
    out = np.array(exps, dtype=np.int64).reshape(-1, 4)
    out.setflags(write=False)
    return out


def _check_resonance_vectors(basis1: Sequence[int], basis2: Sequence[int]) -> None:
    """
    Check that two non-zero resonance vectors are linearly independent.

    Parameters
    ----------
    basis1, basis2 : sequence of 2 int
        Resonance vectors ``[n, m]``.

    Raises
    ------
    ValueError
        If ``basis2`` is non-zero and parallel to ``basis1``.
    """
    if any(basis2) and basis1[0] * basis2[1] - basis1[1] * basis2[0] == 0:
        raise ValueError(
            "The two resonance vectors res_basis1 and res_basis2 must be "
            f"linearly independent, got {list(basis1)} and {list(basis2)}."
        )


def _package_version() -> str | None:
    """
    Installed version of the xnf4d package.

    Returns
    -------
    str or None
        The version, or ``None`` if the package is not installed.
    """
    try:
        return metadata.version("xnf4d")
    except metadata.PackageNotFoundError:
        return None


class NormalForm4D:
    """
    4D normal form of a polynomial one-turn map.

    The class computes the conjugating transformation ``Phi`` and its inverse
    ``Psi`` up to the requested order, the map in normal form ``N``, its
    interpolating vector field ``U`` and the interpolating Hamiltonian ``H``.
    See the module documentation for the conventions.

    Terms of the one-turn map above ``max_nf_order`` do not contribute to
    the normal form up to that order and are ignored.

    A computed normal form can be saved with :meth:`to_json` and reloaded
    with :meth:`from_json` without recomputation, e.g.::

        nf.to_json("nf.json.gz")
        nf = NormalForm4D.from_json("nf.json.gz")

    Parameters
    ----------
    complex_map : Map
        One-turn map in complex normalised Courant-Snyder coordinates
        ``(z1, z1*, z2, z2*)``. Its linear part must be diagonal.
    max_nf_order : int
        Maximum polynomial order of the normal form.
    res_case : {"nonresonant", "resonant", "quasiresonant"}, default "nonresonant"
        Type of normal form:

        - ``"nonresonant"``: all non-linear terms are removed, except those
          depending only on the actions. No resonance vector may be given.
        - ``"resonant"``: the terms of the resonances given by
          ``res_basis1`` (and ``res_basis2``) are kept in the normal form.
          Suited to tunes on or very close to the resonances.
        - ``"quasiresonant"``: as ``"resonant"``, and the normal form is
          expressed in the frame rotating with the resonant tunes given by
          ``res_eig``. Requires ``res_eig``. Not suited to tunes exactly on
          the resonance (use ``"resonant"`` instead).

    res_eig : sequence of 4 complex, optional
        Resonant eigenvalues ``[exp(2i pi Qx_res), exp(-2i pi Qx_res),
        exp(2i pi Qy_res), exp(-2i pi Qy_res)]``. Required in the
        quasi-resonant case only.
    res_basis1 : sequence of 2 int, optional
        ``[n, m]`` satisfying the resonance condition ``n Qx + m Qy = p``.
        Required in the resonant and quasi-resonant cases.
    res_basis2 : sequence of 2 int, optional
        ``[n, m]`` of a second, independent resonance (double resonance).

    Attributes
    ----------
    Phi : Map or None
        Transformation from normal form to Courant-Snyder coordinates.
    Psi : Map or None
        Transformation from Courant-Snyder to normal form coordinates.
    N : Map or None
        One-turn map in normal form, ``Psi o F o Phi``.
    U : Map or None
        Interpolating vector field of ``N``.
    H : Polynom or None
        Interpolating Hamiltonian.
    lin_eig : list of complex
        Eigenvalues of the linear part of the map.
    res_space_dim : int
        Number of resonance vectors given (0, 1 or 2).

    Raises
    ------
    ValueError
        If the parameters are inconsistent.
    """

    def __init__(
        self,
        complex_map: Map,
        max_nf_order: int,
        res_case: ResonanceCase = "nonresonant",
        res_eig: Sequence[complex] | None = None,
        res_basis1: Sequence[int] | None = None,
        res_basis2: Sequence[int] | None = None,
    ) -> None:
        if max_nf_order < 1:
            raise ValueError(f"max_nf_order must be at least 1, got {max_nf_order}.")
        if res_case not in _RES_CASES:
            raise ValueError(
                f"res_case must be one of {_RES_CASES}, got {res_case!r}."
            )
        if res_basis2 is not None and res_basis1 is None:
            raise ValueError("res_basis2 can only be given together with res_basis1.")
        for name, basis in (("res_basis1", res_basis1), ("res_basis2", res_basis2)):
            if basis is not None and (len(basis) != 2 or not any(basis)):
                raise ValueError(
                    f"{name} must be a non-zero resonance vector [n, m], got {basis}."
                )
        if res_case == "nonresonant" and res_basis1 is not None:
            raise ValueError(
                "Resonance vectors cannot be given in the non-resonant case; "
                "use res_case='resonant' or 'quasiresonant'."
            )
        if res_case != "nonresonant" and res_basis1 is None:
            raise ValueError(
                f"The resonance vector res_basis1 must be given in the {res_case} case."
            )
        if res_case == "quasiresonant" and (res_eig is None or len(res_eig) != 4):
            raise ValueError(
                "The 4 resonant eigenvalues res_eig must be given in the "
                f"quasiresonant case, got {res_eig}."
            )
        if res_case != "quasiresonant" and res_eig is not None:
            raise ValueError(
                "res_eig is only used in the quasiresonant case, it must not be "
                f"given in the {res_case} case."
            )

        #: Type of normal form.
        self.res_case: ResonanceCase = res_case
        #: Number of resonance vectors (dimension of the resonant space).
        self.res_space_dim: int = (res_basis1 is not None) + (res_basis2 is not None)
        # Resonance vectors as given by the user (for saving)
        self._res_basis_args = tuple(
            None if b is None else [int(v) for v in b] for b in (res_basis1, res_basis2)
        )
        #: Resonant eigenvalues (quasi-resonant case).
        self.res_eig: list[complex] = (
            [complex(v) for v in res_eig] if res_eig is not None else [0j] * 4
        )

        # The resonance vectors are stored "transposed", as in the original
        # algorithm: res_basis1 = [n of first, n of second resonance] and
        # res_basis2 = [m of first, m of second resonance].
        basis1 = [0, 0]
        basis2 = [0, 0]
        if res_basis1 is not None:
            basis1[0], basis2[0] = int(res_basis1[0]), int(res_basis1[1])
        if res_basis2 is not None:
            basis1[1], basis2[1] = int(res_basis2[0]), int(res_basis2[1])
        _check_resonance_vectors((basis1[0], basis2[0]), (basis1[1], basis2[1]))
        #: First basis array of the resonant space (see above).
        self.res_basis1: list[int] = list(basis1)
        #: Second basis array of the resonant space (see above).
        self.res_basis2: list[int] = list(basis2)

        #: Maximum order of the normal form.
        self.nf_order: int = int(max_nf_order)
        #: Maximum order of the internal coefficient arrays.
        self.nord: int = self.nf_order + 1
        #: Number of coefficients of the internal maps.
        self.N_dim: int = _n_monomials(self.nord)
        #: Number of coefficients of the internal Hamiltonian.
        self.N_dim_H: int = _n_monomials(self.nord + 1)

        self._complex_one_turn_map: Map = complex_map

        # Monomial bookkeeping
        self._exps = _monomial_exponents(self.nord + 1)
        self._orders = self._exps.sum(axis=1)
        base = self.nord + 2
        self._code_strides = np.array([base**3, base**2, base, 1], dtype=np.int64)
        self._codes = self._exps @ self._code_strides
        self._lookup = np.full(base**4, -1, dtype=np.int64)
        self._lookup[self._codes] = np.arange(len(self._exps))
        # _deriv[k, i]: index of monomial i times variable k;
        # _deriv_fac[k, i]: exponent of variable k in that monomial.
        exps = self._exps[: self.N_dim]
        self._deriv = np.empty((4, self.N_dim), dtype=np.int64)
        for k in range(4):
            self._deriv[k] = self._lookup[self._codes[: self.N_dim] + self._code_strides[k]]
        self._deriv_fac = (exps.T + 1).astype(float)

        self._pairs: tuple[np.ndarray, ...] | None = None
        self._selections: dict[tuple[int, ...], tuple[np.ndarray, ...]] = {}

        # One-turn map as coefficient arrays
        self._F0 = self._map_to_array(complex_map)

        #: Eigenvalues of the linear part of the map.
        self.lin_eig: list[complex] = []

        self.Psi: Map | None = None
        self.Phi: Map | None = None
        self.N: Map | None = None
        self.U: Map | None = None
        self.H: Polynom | None = None

        self._reset()

    # ------------------------------------------------------------------ #
    # Set-up helpers
    # ------------------------------------------------------------------ #

    def _map_to_array(self, poly_map: Map) -> np.ndarray:
        """
        Coefficient arrays of a map.

        Coefficients of equal monomials are summed.

        Parameters
        ----------
        poly_map : Map
            Map to convert.

        Returns
        -------
        numpy.ndarray of complex, shape (4, N_dim)
            Coefficients of the four components.

        Terms above order ``nf_order`` are ignored: they do not contribute
        to the normal form up to that order.

        Raises
        ------
        ValueError
            If the map contains negative exponents.
        """
        out = np.zeros((4, self.N_dim), dtype=complex)
        for j, poly in enumerate(poly_map.polys):
            for t in poly.terms:
                if min(t.exponents) < 0:
                    raise ValueError(
                        f"The one-turn map contains the term {t.exponents} with "
                        "a negative exponent."
                    )
                if t.order > self.nf_order:
                    continue
                code = int(np.dot(t.exponents, self._code_strides))
                out[j, self._lookup[code]] += t.coeff
        return out

    def _reset(self) -> None:
        """
        Initialise the internal state for a new calculation.
        """
        shape = (4, self.N_dim)
        # Generating vector field of Phi
        self._A = np.zeros(shape, dtype=complex)
        # One-turn map
        self._F = self._F0.copy()
        # Transformations and normal form vector field
        self._Psi = np.zeros(shape, dtype=complex)
        self._Phi = np.zeros(shape, dtype=complex)
        self._U = np.zeros(shape, dtype=complex)
        self._H = np.zeros(self.N_dim_H, dtype=complex)
        # Intermediate results
        self._Resto1 = np.zeros(shape, dtype=complex)
        self._Resto2 = np.zeros(shape, dtype=complex)

        self.lin_eig = [complex(self._F[j, _LINEAR_IDX[j]]) for j in range(4)]
        # Linear eigenvalues used by the current normalisation step (the
        # detuning lin_eig / res_eig in the second quasi-resonant step).
        self._eig = list(self.lin_eig)
        # Resonance vectors used by the current normalisation step (zero in
        # the second quasi-resonant step)
        self._basis1 = list(self.res_basis1)
        self._basis2 = list(self.res_basis2)
        self._set_identity(self._Phi)

    @staticmethod
    def _set_identity(M: np.ndarray) -> None:
        """
        Set the linear part of a map to the identity, in place.

        Parameters
        ----------
        M : numpy.ndarray of complex, shape (4, N)
            Coefficient arrays of the map.
        """
        for j in range(4):
            M[j, _LINEAR_IDX[j]] = 1.0

    def _array_to_polynom(self, coeffs: np.ndarray) -> Polynom:
        """
        Polynomial from a coefficient array, without the zero terms.

        Parameters
        ----------
        coeffs : numpy.ndarray of complex
            Coefficient array.

        Returns
        -------
        Polynom
            Polynomial with one term per non-zero coefficient, in index
            order.
        """
        idx = np.flatnonzero(np.abs(coeffs) > _ZERO_ATOL)
        exps = self._exps[idx].tolist()
        return Polynom(
            terms=[Term(c, *e) for c, e in zip(coeffs[idx].tolist(), exps)]
        )

    def _array_to_map(self, M: np.ndarray) -> Map:
        """
        Map from coefficient arrays.

        Parameters
        ----------
        M : numpy.ndarray of complex, shape (4, N)
            Coefficient arrays of the four components.

        Returns
        -------
        Map
            The corresponding map.
        """
        return Map(*(self._array_to_polynom(M[j]) for j in range(4)))

    # ------------------------------------------------------------------ #
    # Products of polynomials
    # ------------------------------------------------------------------ #

    def _all_pairs(self) -> tuple[np.ndarray, ...]:
        """
        All pairs of monomials whose product has order at most ``nf_order``.

        Pairs are sorted by product monomial, then by the first factor, which
        is the order of summation of the original algorithm.

        Returns
        -------
        out, p, q, out_order, p_order : numpy.ndarray of int
            Index of the product, of the two factors, and orders of the
            product and of the first factor.
        """
        if self._pairs is None:
            max_order = self.nf_order
            n = _n_monomials(max_order)
            counts = np.array(
                [_n_monomials(max_order - o) for o in self._orders[:n]], dtype=np.int64
            )
            p = np.repeat(np.arange(n), counts)
            starts = np.cumsum(counts) - counts
            q = np.arange(counts.sum()) - np.repeat(starts, counts)
            out = self._lookup[self._codes[p] + self._codes[q]]
            order = np.lexsort((p, out))
            out, p, q = out[order], p[order], q[order]
            self._pairs = (out, p, q, self._orders[out], self._orders[p])
        return self._pairs

    def _pair_selection(
        self,
        init: tuple[int, int, int],
        max_q: int,
        min_p: int,
        min_r: int,
        max_r: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Pairs of monomials contributing to a truncated product.

        The product ``R = Q * P`` is evaluated for orders ``min_r`` to
        ``max_r`` of ``R``, using the orders of ``P`` allowed by the address
        tables ``init = (max_q, max_p, max_r)`` and at least ``min_p``.

        Parameters
        ----------
        init : tuple of 3 int
            Order limits ``(max_q, max_p, max_r)`` of the address tables.
        max_q : int
            Maximum order of ``Q``.
        min_p : int
            Minimum order of ``P``.
        min_r : int
            Minimum order of ``R``.
        max_r : int
            Maximum order of ``R``.

        Returns
        -------
        flat_out : numpy.ndarray of int
            Index of the product for the 4 components stacked (component
            ``j`` at offset ``j * n_out``).
        p, q : numpy.ndarray of int
            Indices of the factors of ``P`` and ``Q``.
        n_out : int
            Number of coefficients of ``R`` that are computed.
        """
        key = (*init, max_q, min_p, min_r, max_r)
        if key not in self._selections:
            init_q, init_p, init_r = init
            mq = min(init_r, init_q)
            mp = min(init_r, init_p)
            out, p, q, o, po = self._all_pairs()
            lo = np.maximum(0, o - mq)
            hi = np.minimum(o, mp)
            min_pp = np.maximum(min_p, o - max_q)
            # Faithful to the original address arrays: an out-of-range start
            # order points to an unset (zero) entry, i.e. to the first pair.
            start = np.where(min_pp > hi + 1, lo, np.maximum(lo, min_pp))
            mask = (
                (o >= min_r) & (o <= max_r) & (o <= init_r) & (po >= start) & (po <= hi)
            )
            n_out = _last(max_r) + 1
            out_sel = out[mask]
            flat_out = (out_sel[None, :] + n_out * np.arange(4)[:, None]).ravel()
            self._selections[key] = (flat_out, p[mask], q[mask], n_out)
        return self._selections[key]

    @staticmethod
    def _accumulate(flat_out: np.ndarray, values: np.ndarray, n_out: int) -> np.ndarray:
        """
        Sum products into the coefficients they contribute to.

        Parameters
        ----------
        flat_out : numpy.ndarray of int
            Target index of every value (see :meth:`_pair_selection`).
        values : numpy.ndarray of complex, shape (4, n_pairs)
            Products to sum.
        n_out : int
            Number of coefficients per component.

        Returns
        -------
        numpy.ndarray of complex, shape (4, n_out)
            Summed coefficients.
        """
        v = values.ravel()
        res = np.empty(4 * n_out, dtype=complex)
        res.real = np.bincount(flat_out, weights=v.real, minlength=4 * n_out)
        res.imag = np.bincount(flat_out, weights=v.imag, minlength=4 * n_out)
        return res.reshape(4, n_out)

    def _lie_derivative(
        self,
        A: np.ndarray,
        C: np.ndarray,
        R: np.ndarray,
        n_a: int,
        min_c: int,
        n_c: int,
        min_r: int,
        max_r: int,
        init: tuple[int, int, int],
    ) -> np.ndarray:
        """
        Lie derivative of the map ``C`` along the vector field ``A``.

        Computes ``R_j = sum_k A_k dC_j/dz_k`` for orders ``min_r`` to
        ``max_r``.

        Parameters
        ----------
        A : numpy.ndarray of complex, shape (4, N_dim)
            Vector field, used up to order ``n_a``.
        C : numpy.ndarray of complex, shape (4, N_dim)
            Map to differentiate, used up to order ``n_c``.
        R : numpy.ndarray of complex, shape (4, N_dim)
            Output array. Coefficients up to order ``max_r`` are
            overwritten (those below ``min_r`` are set to zero), the others
            are left unchanged.
        n_a : int
            Maximum order of ``A``.
        min_c : int
            Minimum order of ``C``.
        n_c : int
            Maximum order of ``C``.
        min_r : int
            Minimum order of the result.
        max_r : int
            Maximum order of the result.
        init : tuple of 3 int
            Order limits of the product address tables.

        Returns
        -------
        numpy.ndarray of complex, shape (4, N_dim)
            The array ``R``.
        """
        flat_out, p, q, n_out = self._pair_selection(init, n_a, min_c - 1, min_r, max_r)
        m = _n_monomials(n_c - 1)
        D = np.zeros((4, self.N_dim), dtype=complex)
        total = None
        for k in range(4):
            D[:, :m] = self._deriv_fac[k, :m] * C[:, self._deriv[k, :m]]
            B = self._accumulate(flat_out, A[k, q] * D[:, p], n_out)
            total = B if total is None else total + B
        R[:, :n_out] = total
        return R

    def _lie_transformation(
        self,
        A: np.ndarray,
        S: np.ndarray,
        C: np.ndarray,
        max_a: int,
        max_r: int,
    ) -> np.ndarray:
        """
        Lie transformation ``C = exp(L_A) S`` up to order ``max_r``.

        Parameters
        ----------
        A : numpy.ndarray of complex, shape (4, N_dim)
            Vector field, used up to order ``max_a``.
        S : numpy.ndarray of complex, shape (4, N_dim)
            Map to transform.
        C : numpy.ndarray of complex, shape (4, N_dim)
            Output array. Coefficients up to order ``max_r`` are
            overwritten, the others are left unchanged.
        max_a : int
            Maximum order of ``A``.
        max_r : int
            Maximum order of the result.

        Returns
        -------
        numpy.ndarray of complex, shape (4, N_dim)
            The array ``C``.
        """
        n = _last(max_r) + 1
        Q = np.zeros((4, self.N_dim), dtype=complex)
        Q[:, :n] = S[:, :n]
        C[:, :n] = Q[:, :n]
        R = np.zeros((4, self.N_dim), dtype=complex)
        init = (max_a, max_r - 1, max_r)
        for nord in range(1, max_r):
            R = self._lie_derivative(
                A, Q, R, max_a, nord, max_r, nord + 1, max_r, init
            )
            Q[:, :n] = R[:, :n] / nord
            C[:, :n] = C[:, :n] + Q[:, :n]
        return C

    @staticmethod
    def _lin_eval_factors(vec: Sequence[complex], max_order: int) -> np.ndarray:
        """
        Values ``vec[0]**n1 vec[1]**n2 vec[2]**n3 vec[3]**n4`` of all monomials.

        Parameters
        ----------
        vec : sequence of 4 complex
            Values of the four variables.
        max_order : int
            Maximum order of the monomials.

        Returns
        -------
        numpy.ndarray of complex
            Value of every monomial up to ``max_order``, in index order
            (index ``0``, the constant, is set to ``1``).
        """
        out = [1.0 + 0j]
        rap14 = vec[0] / vec[3]
        rap24 = vec[1] / vec[3]
        rap34 = vec[2] / vec[3]
        eval_ = 1.0 + 0j
        for nord in range(1, max_order + 1):
            eval_ = vec[3] * eval_
            eval1 = eval_
            for n1 in range(nord + 1):
                eval2 = eval1
                for n2 in range(nord - n1 + 1):
                    eval3 = eval2
                    for _ in range(nord - n1 - n2 + 1):
                        out.append(eval3)
                        eval3 *= rap34
                    eval2 *= rap24
                eval1 *= rap14
        return np.array(out, dtype=complex)

    @staticmethod
    def _comp_lin(
        P: np.ndarray, vec: Sequence[complex], Q: np.ndarray, max_order: int
    ) -> np.ndarray:
        """
        Composition ``Q = P o diag(vec)`` with a diagonal linear map.

        Parameters
        ----------
        P : numpy.ndarray of complex, shape (4, N_dim)
            Map to compose.
        vec : sequence of 4 complex
            Diagonal of the linear map.
        Q : numpy.ndarray of complex, shape (4, N_dim)
            Output array. Coefficients of orders ``1`` to ``max_order`` are
            overwritten, the others are left unchanged.
        max_order : int
            Maximum order of the result.

        Returns
        -------
        numpy.ndarray of complex, shape (4, N_dim)
            The array ``Q``.
        """
        n = _n_monomials(max_order)
        factors = NormalForm4D._lin_eval_factors(vec, max_order)
        Q[:, 1:n] = P[:, 1:n] * factors[1:n]
        return Q

    @staticmethod
    def _hom_eq_factors(vec: Sequence[complex], order: int) -> np.ndarray:
        """
        Values ``vec[0]**n1 vec[1]**n2 vec[2]**n3 vec[3]**n4`` of one order.

        Same quantity as :meth:`_lin_eval_factors`, computed as in the
        homological equation of the original algorithm.

        Parameters
        ----------
        vec : sequence of 4 complex
            Values of the four variables.
        order : int
            Total order of the monomials.

        Returns
        -------
        numpy.ndarray of complex
            Value of every monomial of the given order, in index order.
        """
        out = []
        pd1 = 1.0 + 0j
        pd4 = vec[3] ** order
        rap14 = vec[0] / vec[3]
        rap24 = vec[1] / vec[3]
        rap34 = vec[2] / vec[3]
        for n1 in range(order + 1):
            pd2 = pd1
            for n2 in range(order - n1 + 1):
                pd3 = pd2
                for _ in range(order - n1 - n2 + 1):
                    out.append(pd3 * pd4)
                    pd3 = pd3 * rap34
                pd2 = pd2 * rap24
            pd1 = pd1 * rap14
        return np.array(out, dtype=complex)

    @staticmethod
    def fmod(a: int, p: int) -> int:
        """
        Remainder of an integer division, truncated towards zero.

        Parameters
        ----------
        a : int
            Dividend.
        p : int
            Divisor.

        Returns
        -------
        int
            ``a - int(a / p) * p`` (same sign as ``a``, as Fortran ``MOD``).
        """
        return a - (int(a / p) * p)

    def _resonant_mask(self, k1: np.ndarray, k2: np.ndarray) -> np.ndarray:
        """
        Which exponent combinations are resonant.

        Parameters
        ----------
        k1, k2 : numpy.ndarray of int
            Exponent differences ``n1 - n2`` and ``n3 - n4`` of the terms,
            shifted for the map component.

        Returns
        -------
        numpy.ndarray of bool
            ``True`` where ``(k1, k2)`` is an integer combination of the
            resonance vectors.
        """
        b1, b2 = self._basis1, self._basis2
        i_norm1 = b1[0] ** 2 + b2[0] ** 2
        i_norm2 = b1[1] ** 2 + b2[1] ** 2
        i_prodv = b1[0] * b2[1] - b2[0] * b1[1]
        if i_norm2 != 0:
            i_prodv1 = k1 * b2[0] - k2 * b1[0]
            i_prodv2 = k1 * b2[1] - k2 * b1[1]
            i_mod2 = np.fmod(-i_prodv1, i_prodv)
            i_mod1 = np.fmod(i_prodv2, i_prodv)
        elif i_norm1 != 0:
            iscal1 = k1 * b1[0] + k2 * b2[0]
            i_mod2 = k1 * b2[0] - k2 * b1[0]
            i_mod1 = np.fmod(iscal1, i_norm1)
        else:
            i_mod1 = i_mod2 = np.ones_like(k1)
        return ((k1 == 0) & (k2 == 0)) | ((i_mod1 == 0) & (i_mod2 == 0))

    def _solve_hom_eq(
        self,
        Phi: np.ndarray,
        U: np.ndarray,
        Resto: np.ndarray,
        max_r: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve the homological equation at order ``max_r``.

        Resonant terms of the remainder go to the normal form vector field
        ``U``, the others are removed by the generating field ``A``.

        Parameters
        ----------
        Phi : numpy.ndarray of complex, shape (4, N_dim)
            Conjugating transformation, updated in place.
        U : numpy.ndarray of complex, shape (4, N_dim)
            Normal form vector field, updated in place.
        Resto : numpy.ndarray of complex, shape (4, N_dim)
            Remainder of the functional equation.
        max_r : int
            Order of the equation.

        Returns
        -------
        Phi, U, A : numpy.ndarray of complex, shape (4, N_dim)
            Updated transformation, normal form vector field and generating
            vector field.

        Raises
        ------
        ZeroDivisionError
            If a non-resonant term has an exactly vanishing denominator.
        """
        lo, hi = _first(max_r), _last(max_r) + 1
        n1, n2, n3, n4 = self._exps[lo:hi].T
        lam = self._eig
        pd = self._hom_eq_factors(lam, max_r)
        for j in range(4):
            k1 = n1 - n2 - _K_SHIFT[0, j]
            k2 = n3 - n4 - _K_SHIFT[1, j]
            res = self._resonant_mask(k1, k2)
            rest = Resto[j, lo:hi]
            pden = pd - lam[j]
            nonres = ~res
            with np.errstate(divide="ignore", invalid="ignore"):
                a = np.where(rest == 0, 0, rest / pden)
            if np.any(nonres & (rest != 0) & (pden == 0)):
                raise ZeroDivisionError(
                    f"Vanishing denominator at order {max_r}: the map is exactly "
                    "resonant but the resonance is not included in the "
                    "resonant basis."
                )
            self._A[j, lo:hi] = np.where(res, 0, a)
            U[j, lo:hi] = np.where(res, rest / lam[j], 0)
            Phi[j, lo:hi] = np.where(res, Phi[j, lo:hi], Phi[j, lo:hi] + a)
        return Phi, U, self._A

    def _inverse(
        self,
        Psi: np.ndarray,
        max_r: int,
        A: np.ndarray,
    ) -> np.ndarray:
        """
        Inverse ``Psi = exp(L_{-A}) id`` of the transformation ``Phi``.

        Parameters
        ----------
        Psi : numpy.ndarray of complex, shape (4, N_dim)
            Output array, overwritten up to order ``max_r``.
        max_r : int
            Maximum order of the result.
        A : numpy.ndarray of complex, shape (4, N_dim)
            Generating vector field of ``Phi``. It is negated in place.

        Returns
        -------
        numpy.ndarray of complex, shape (4, N_dim)
            The array ``Psi``.
        """
        n = _last(max_r) + 1
        C = np.zeros((4, self.N_dim), dtype=complex)
        R = np.zeros((4, self.N_dim), dtype=complex)
        Psi[:, :n] = 0
        A[:, :n] = -A[:, :n]
        self._set_identity(C)
        self._set_identity(Psi)
        init = (max_r, max_r - 1, max_r)
        for nord in range(1, max_r):
            R = self._lie_derivative(A, C, R, max_r, nord, max_r, nord + 1, max_r, init)
            C[:, :n] = R[:, :n] / nord
            Psi[:, 5:n] += C[:, 5:n]
        return Psi

    def _hamiltonian(self, A: np.ndarray, H: np.ndarray, max_r: int) -> np.ndarray:
        """
        Interpolating Hamiltonian of the vector field ``A``.

        The linear part ``log(lambda_j)`` (eigenvalues of the current
        normalisation step) is first written into ``A``. The
        Hamiltonian then satisfies ``A_0 = dH/dz1*``, ``A_1 = -dH/dz1``,
        ``A_2 = dH/dz2*`` and ``A_3 = -dH/dz2``.

        Parameters
        ----------
        A : numpy.ndarray of complex, shape (4, N_dim)
            Vector field; its linear part is set in place.
        H : numpy.ndarray of complex, shape (N_dim_H,)
            Output array.
        max_r : int
            Maximum order of ``A`` that is integrated.

        Returns
        -------
        numpy.ndarray of complex, shape (N_dim_H,)
            The array ``H``.
        """
        for j in range(4):
            A[j, _LINEAR_IDX[j]] = np.log(self._eig[j])
        src = np.arange(_first(1), _last(max_r) + 1)
        e = self._exps[src]
        # Each step overwrites the previous one where both define the same
        # coefficient, in the same order as the original algorithm.
        H[self._deriv[0, src]] = -A[1, src] / (e[:, 0] + 1)
        s = src[e[:, 0] == 0]
        H[self._deriv[1, s]] = A[0, s] / (self._exps[s, 1] + 1)
        s = src[(e[:, 0] == 0) & (e[:, 1] == 0)]
        H[self._deriv[2, s]] = -A[3, s] / (self._exps[s, 2] + 1)
        s = src[(e[:, 0] == 0) & (e[:, 1] == 0) & (e[:, 2] == 0)]
        H[self._deriv[3, s]] = A[2, s] / (self._exps[s, 3] + 1)
        return H

    def _normalise(self) -> None:
        """
        Order-by-order normalisation loop (orders 2 to ``nf_order``).
        """
        for nord in range(2, self.nf_order + 1):
            n_phi = nord - 1
            n_u = nord - 1

            # Write Phi as a Lie transformation up to order nord
            n = _last(nord) + 1
            self._Psi[:, :n] = 0
            self._set_identity(self._Psi)
            self._Phi = self._lie_transformation(self._A, self._Psi, self._Phi, n_phi, nord)

            # Lie transformation by A of the map F up to order nord
            self._Resto1 = self._lie_transformation(
                self._A, self._F, self._Resto1, n_phi, nord
            )

            # Lie transformation of Phi by the interpolating field U of the
            # normal form: first compose with the linear part, then apply
            # the Lie transformation of the non-linear part
            self._Psi = self._comp_lin(self._Phi, self._eig, self._Psi, nord)
            self._Resto2 = self._lie_transformation(
                self._U, self._Psi, self._Resto2, n_u, nord
            )

            # Remainder of the functional equation at order nord
            lo, hi = _first(nord), _n_monomials(nord) + 1
            self._Resto1[:, lo:hi] = self._Resto1[:, lo:hi] - self._Resto2[:, lo:hi]

            # Solve the homological equation, to compute U, A and Phi
            self._Phi, self._U, self._A = self._solve_hom_eq(
                self._Phi, self._U, self._Resto1, nord
            )

    def compute_normal_form(self) -> None:
        """
        Compute the normal form.

        Sets the attributes :attr:`Phi`, :attr:`Psi`, :attr:`N`, :attr:`U`
        and :attr:`H`. Calling the method again recomputes everything from
        the one-turn map.
        """
        self._reset()
        nf = self.nf_order

        self._normalise()
        self._Psi = self._inverse(self._Psi, nf, self._A)

        self.Phi = self._array_to_map(self._Phi)
        self.Psi = self._array_to_map(self._Psi)

        n_coef = _last(nf) + 1

        # Map in normal form: N = Lambda o flow(U), with U (no linear part
        # at this point) the field of the first normalisation
        identity = np.zeros((4, self.N_dim), dtype=complex)
        self._set_identity(identity)
        flow = self._lie_transformation(
            self._U, identity, np.zeros_like(identity), nf, nf
        )
        self.N = self._array_to_map(flow * np.array(self._eig)[:, None])

        if self.res_case == "quasiresonant":
            # Normal form map associated to the field U, stored in Phi for
            # the moment
            self._Psi[:, :n_coef] = 0
            self._set_identity(self._Psi)
            self._Phi = self._lie_transformation(self._U, self._Psi, self._Phi, nf, nf)

            # Redefine the map F for the resonant case
            for j in range(4):
                ratio = self._eig[j] / self.res_eig[j]
                self._F[j, :n_coef] = self._Phi[j, :n_coef] * ratio
            self._U[:, :n_coef] = 0
            self._Phi[:, :n_coef] = 0
            self._A[:, :n_coef] = 0
            self._basis1 = [0, 0]
            self._basis2 = [0, 0]

            # Initialise Phi and U and compute the non-resonant normal form
            self._eig = [complex(self._F[j, _LINEAR_IDX[j]]) for j in range(4)]
            self._U[:, :5] = 0
            self._Phi[:, :5] = 0
            self._A[:, 1:5] = 0
            self._set_identity(self._Phi)
            self._normalise()

            # Change the sign of A to return to the resonant variables by
            # means of a Lie transformation
            self._A[:, 1:n_coef] = -self._A[:, 1:n_coef]
            # Add the linear part to the field U
            for j in range(4):
                self._U[j, _LINEAR_IDX[j]] = np.log(self._eig[j])
            # Back to resonant variables
            self._Psi = self._lie_derivative(
                self._U, self._Phi, self._Psi, nf, 1, nf, 1, nf, (nf, nf - 1, nf)
            )
            self._U = self._lie_transformation(self._A, self._Psi, self._U, nf, nf)

        self._H = self._hamiltonian(self._U, self._H, nf + 1)

        self.U = self._array_to_map(self._U)
        self.H = self._array_to_polynom(self._H)

    # ------------------------------------------------------------------ #
    # Saving and loading
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        """
        Representation of the normal form with plain Python types.

        The result can be written to JSON (see :meth:`to_json`) and read
        back with :meth:`from_dict` without recomputing the normal form.
        Floats are stored exactly.

        Returns
        -------
        dict
            Dictionary with the keys:

            - ``"format"``, ``"format_version"``, ``"xnf4d_version"``:
              identification of the file format and of the package version;
            - ``"settings"``: the constructor arguments ``max_nf_order``,
              ``res_case``, ``res_eig``, ``res_basis1``, ``res_basis2``
              (complex numbers are stored as ``[real, imag]``);
            - ``"lin_eig"``: eigenvalues of the linear part of the map, for
              information;
            - ``"complex_map"``: the one-turn map, truncated at
              ``max_nf_order`` (see :meth:`Map.to_dict`);
            - ``"results"``: ``None`` if the normal form has not been
              computed, otherwise the maps ``"Phi"``, ``"Psi"``, ``"N"``,
              ``"U"`` and the polynomial ``"H"``.
        """
        res_eig = (
            [[v.real, v.imag] for v in self.res_eig]
            if self.res_case == "quasiresonant"
            else None
        )
        results = None
        if self.H is not None:
            results = {
                name: getattr(self, name).to_dict() for name in ("Phi", "Psi", "N", "U", "H")
            }
        return {
            "format": _FORMAT_NAME,
            "format_version": _FORMAT_VERSION,
            "xnf4d_version": _package_version(),
            "settings": {
                "max_nf_order": self.nf_order,
                "res_case": self.res_case,
                "res_eig": res_eig,
                "res_basis1": self._res_basis_args[0],
                "res_basis2": self._res_basis_args[1],
            },
            "lin_eig": [[v.real, v.imag] for v in self.lin_eig],
            "complex_map": self._array_to_map(self._F0).to_dict(),
            "results": results,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NormalForm4D:
        """
        Rebuild a normal form from the output of :meth:`to_dict`.

        The results are restored as saved, without recomputation; the
        stored ``"lin_eig"`` is ignored, as it follows from the map.

        Parameters
        ----------
        data : dict
            Output of :meth:`to_dict`.

        Returns
        -------
        NormalForm4D
            The normal form. If the saved normal form had been computed, its
            attributes ``Phi``, ``Psi``, ``N``, ``U`` and ``H`` are set.

        Raises
        ------
        ValueError
            If the data does not describe a normal form, or was written by
            a newer, unsupported version of the format.
        """
        if not isinstance(data, dict) or data.get("format") != _FORMAT_NAME:
            raise ValueError("The data does not describe an xnf4d NormalForm4D.")
        version = data.get("format_version")
        if not isinstance(version, int) or not 1 <= version <= _FORMAT_VERSION:
            raise ValueError(
                f"Unsupported NormalForm4D format version {version!r}; this "
                f"version of xnf4d reads versions up to {_FORMAT_VERSION}."
            )
        try:
            settings = data["settings"]
            res_eig = settings["res_eig"]
            nf = cls(
                Map.from_dict(data["complex_map"]),
                settings["max_nf_order"],
                settings["res_case"],
                res_eig=None if res_eig is None else [complex(*v) for v in res_eig],
                res_basis1=settings["res_basis1"],
                res_basis2=settings["res_basis2"],
            )
            results = data["results"]
        except (KeyError, TypeError) as err:
            raise ValueError(f"Invalid NormalForm4D data: {err!r}.") from None
        if results is not None:
            try:
                for name in ("Phi", "Psi", "N", "U"):
                    setattr(nf, name, Map.from_dict(results[name]))
                nf.H = Polynom.from_dict(results["H"])
            except (KeyError, TypeError) as err:
                raise ValueError(f"Invalid NormalForm4D results: {err!r}.") from None
        return nf

    def to_json(self, path: str | os.PathLike[str], indent: int | None = None) -> None:
        """
        Save the normal form to a JSON file.

        Parameters
        ----------
        path : str or os.PathLike
            File to write. If the name ends with ``.gz`` the file is
            compressed with gzip.
        indent : int, optional
            Indentation passed to :func:`json.dump`. By default the file is
            written without line breaks, which keeps it small.
        """
        opener = gzip.open if os.fspath(path).endswith(".gz") else open
        with opener(path, "wt", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=indent)

    @classmethod
    def from_json(cls, path: str | os.PathLike[str]) -> NormalForm4D:
        """
        Load a normal form saved with :meth:`to_json`.

        Gzip-compressed files are recognised automatically, whatever their
        name.

        Parameters
        ----------
        path : str or os.PathLike
            File to read.

        Returns
        -------
        NormalForm4D
            The normal form (see :meth:`from_dict`).

        Raises
        ------
        ValueError
            If the file does not contain a supported normal form.
        """
        with open(path, "rb") as f:
            compressed = f.read(2) == b"\x1f\x8b"
        opener = gzip.open if compressed else open
        with opener(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    # ------------------------------------------------------------------ #
    # Coordinate transformations
    # ------------------------------------------------------------------ #

    def norm_to_nf(
        self,
        z1: ArrayOrScalar,
        z1s: ArrayOrScalar,
        z2: ArrayOrScalar,
        z2s: ArrayOrScalar,
    ) -> tuple[ArrayOrScalar, ArrayOrScalar, ArrayOrScalar, ArrayOrScalar]:
        """
        Transform complex Courant-Snyder coordinates to normal form ones.

        Use :func:`real_to_complex` to obtain the complex coordinates from
        real ones.

        Parameters
        ----------
        z1, z1s, z2, z2s : complex or numpy.ndarray
            Complex Courant-Snyder coordinates ``(z1, z1*, z2, z2*)``.

        Returns
        -------
        tuple of 4 complex or numpy.ndarray
            Normal form coordinates ``(zeta1, zeta1*, zeta2, zeta2*)``.

        Raises
        ------
        ValueError
            If the normal form has not been computed.
        """
        if self.Psi is None:
            raise ValueError("Conjugating function has not been evaluated!")
        return self.Psi.substitute(z1, z1s, z2, z2s)

    def nf_to_norm(
        self,
        zeta1: ArrayOrScalar,
        zeta1s: ArrayOrScalar,
        zeta2: ArrayOrScalar,
        zeta2s: ArrayOrScalar,
    ) -> tuple[ArrayOrScalar, ArrayOrScalar, ArrayOrScalar, ArrayOrScalar]:
        """
        Transform normal form coordinates to complex Courant-Snyder ones.

        Use :func:`complex_to_real` to obtain real coordinates from the
        result.

        Parameters
        ----------
        zeta1, zeta1s, zeta2, zeta2s : complex or numpy.ndarray
            Normal form coordinates ``(zeta1, zeta1*, zeta2, zeta2*)``.

        Returns
        -------
        tuple of 4 complex or numpy.ndarray
            Complex Courant-Snyder coordinates ``(z1, z1*, z2, z2*)``.

        Raises
        ------
        ValueError
            If the normal form has not been computed.
        """
        if self.Phi is None:
            raise ValueError("Conjugating function has not been evaluated!")
        return self.Phi.substitute(zeta1, zeta1s, zeta2, zeta2s)
