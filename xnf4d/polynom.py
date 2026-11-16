"""
Truncated polynomial algebra in four variables.

This module provides the building blocks for 4D normal form computations:
:class:`Term`, :class:`Polynom` and :class:`Map` for polynomials and
polynomial maps with numerical coefficients.

The four variables are not tied to a specific coordinate system. They are
referred to by their index ``0``, ``1``, ``2`` and ``3`` (for example
``x, px, y, py`` or ``zeta_x, zeta_x^*, zeta_y, zeta_y^*``).

Two numerical tolerances are used throughout:

- a coefficient ``c`` is treated as zero when ``abs(c) <= 1e-16``
  (equivalent to ``np.isclose(c, 0, rtol=1e-14, atol=1e-16)``);
- relative tolerance ``1e-14`` is used when comparing coefficients.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Union

import numpy as np

__all__ = [
    "NUM_VARS",
    "Coeff",
    "Exponents",
    "ArrayOrScalar",
    "Term",
    "Polynom",
    "Map",
]

#: Number of variables of the polynomials.
NUM_VARS = 4

#: Numerical coefficient of a term.
Coeff = Union[int, float, complex]
#: Exponents of the four variables of a term.
Exponents = tuple[int, int, int, int]
#: Value that can be substituted for a variable.
ArrayOrScalar = Union[float, complex, np.ndarray]

_RTOL = 1e-14
_ATOL = 1e-16
_ZERO_EXPS: Exponents = (0, 0, 0, 0)


# --------------------------------------------------------------------------- #
# Scalar helpers (fast replacements of ``np.isclose`` for Python scalars)
# --------------------------------------------------------------------------- #


def _is_zero(value: Any) -> bool:
    """
    Check whether a coefficient can be considered zero.

    Same result as ``np.isclose(value, 0, rtol=1e-14, atol=1e-16)`` for a
    scalar, but much faster.

    Parameters
    ----------
    value : int, float or complex
        Value to check.

    Returns
    -------
    bool
        ``True`` if ``abs(value) <= 1e-16``.
    """
    return bool(abs(value) <= _ATOL)


def _isclose(a: float, b: float, rtol: float, atol: float) -> bool:
    """
    Scalar version of ``np.isclose(a, b, rtol=rtol, atol=atol)``.

    Parameters
    ----------
    a, b : float
        Values to compare. As in numpy, the test is not symmetric:
        ``abs(a - b) <= atol + rtol * abs(b)``.
    rtol : float
        Relative tolerance.
    atol : float
        Absolute tolerance.

    Returns
    -------
    bool
        ``True`` if the values are close.
    """
    if a == b:
        return True
    if not (math.isfinite(a) and math.isfinite(b)):
        return False
    return abs(a - b) <= atol + rtol * abs(b)


def _order_of_magnitude(value: float) -> int:
    """
    Decimal order of magnitude of a value, or ``0`` if the value is zero.

    Parameters
    ----------
    value : float
        Value to analyse.

    Returns
    -------
    int
        ``floor(log10(abs(value)))``, or ``0`` if ``value`` is zero within
        tolerance.
    """
    if abs(value) <= _ATOL:
        return 0
    return math.floor(math.log10(abs(value)))


@lru_cache(maxsize=None)
def _power_of_ten(exponent: int) -> float:
    """
    Return ``10**exponent`` exactly as the float literal ``1e<exponent>``.

    Parameters
    ----------
    exponent : int
        Decimal exponent.

    Returns
    -------
    float
        The value ``float(f"1e{exponent}")``.
    """
    return float(f"1e{exponent}")


def _check_var(var: int) -> None:
    """
    Check that ``var`` is a valid variable index.

    Parameters
    ----------
    var : int
        Variable index, must be one of ``0, 1, 2, 3``.

    Raises
    ------
    ValueError
        If ``var`` is not a valid variable index.
    """
    if (
        isinstance(var, bool)
        or not isinstance(var, (int, np.integer))
        or not 0 <= var < NUM_VARS
    ):
        raise ValueError(
            f"Incorrect variable {var!r}: expected an integer index in "
            f"[0, {NUM_VARS - 1}]."
        )


def _check_power(pow: int) -> None:
    """
    Check that ``pow`` is a valid (non-negative) power.

    Parameters
    ----------
    pow : int
        Power to check.

    Raises
    ------
    ValueError
        If ``pow`` is negative.
    """
    if pow < 0:
        raise ValueError(f"Power must be a non-negative integer, got {pow}.")


def _power_tables(
    values: Sequence[ArrayOrScalar],
    exponents: Iterable[Exponents],
) -> list[dict[int, ArrayOrScalar]]:
    """
    Pre-compute ``values[i] ** k`` for all exponents ``k`` that are needed.

    Each power is computed once with the same expression as a direct
    evaluation, so results are identical, only faster.

    Parameters
    ----------
    values : sequence of 4 floats, complex numbers or arrays
        Values of the four variables.
    exponents : iterable of 4-tuples of int
        Exponents that will be used.

    Returns
    -------
    list of 4 dict
        ``tables[i][k] == values[i] ** k``.
    """
    needed: list[set[int]] = [set(), set(), set(), set()]
    for exps in exponents:
        for i in range(NUM_VARS):
            needed[i].add(exps[i])
    return [{k: values[i] ** k for k in needed[i]} for i in range(NUM_VARS)]


# --------------------------------------------------------------------------- #
# Numerical terms, polynomials and maps
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Term:
    """
    A monomial of 4 variables with a numerical coefficient.

    The term represents ``coeff * v0**exp0 * v1**exp1 * v2**exp2 * v3**exp3``.

    Parameters
    ----------
    coeff : int, float or complex
        Numerical value of the coefficient.
    exp0 : int, default 0
        Exponent of the first variable.
    exp1 : int, default 0
        Exponent of the second variable.
    exp2 : int, default 0
        Exponent of the third variable.
    exp3 : int, default 0
        Exponent of the fourth variable.
    """

    coeff: Coeff
    exp0: int = 0
    exp1: int = 0
    exp2: int = 0
    exp3: int = 0

    @property
    def exponents(self) -> Exponents:
        """
        Exponents of the four variables.

        Returns
        -------
        tuple of 4 int
            ``(exp0, exp1, exp2, exp3)``.
        """
        return (self.exp0, self.exp1, self.exp2, self.exp3)

    @property
    def order(self) -> int:
        """
        Total order of the term.

        Returns
        -------
        int
            ``exp0 + exp1 + exp2 + exp3``.
        """
        return self.exp0 + self.exp1 + self.exp2 + self.exp3

    def __str__(self) -> str:
        return f"{self.coeff}\t( {self.exp0} {self.exp1} {self.exp2} {self.exp3} )"

    def __eq__(self, other: object) -> bool:
        """
        Compare two terms.

        Exponents must be identical. The real and imaginary parts of the
        coefficients are compared with a relative tolerance of ``1e-14`` and
        an absolute tolerance 10 orders of magnitude below the smaller of the
        two values.

        Parameters
        ----------
        other : object
            Object to compare with.

        Returns
        -------
        bool
            ``True`` if the terms are equal within tolerance. Comparison
            with an object that is not a :class:`Term` returns
            ``NotImplemented`` (so ``==`` gives ``False``).
        """
        if not isinstance(other, Term):
            return NotImplemented

        if (
            self.exp0 != other.exp0
            or self.exp1 != other.exp1
            or self.exp2 != other.exp2
            or self.exp3 != other.exp3
        ):
            return False

        s_re, s_im = self.coeff.real, self.coeff.imag
        o_re, o_im = other.coeff.real, other.coeff.imag

        re_atol = _power_of_ten(
            min(_order_of_magnitude(s_re), _order_of_magnitude(o_re)) - 10
        )
        if not _isclose(s_re, o_re, _RTOL, re_atol):
            return False

        im_atol = _power_of_ten(
            min(_order_of_magnitude(s_im), _order_of_magnitude(o_im)) - 10
        )
        return _isclose(s_im, o_im, _RTOL, im_atol)

    def __hash__(self) -> int:
        rounded_coeff = complex(round(self.coeff.real, 14), round(self.coeff.imag, 14))
        return hash((rounded_coeff, self.exp0, self.exp1, self.exp2, self.exp3))

    @staticmethod
    def product_Terms(term1: Term, term2: Term, max_order: int) -> Term:
        """
        Product of two terms, truncated at a given order.

        Parameters
        ----------
        term1 : Term
            First term in the product.
        term2 : Term
            Second term in the product.
        max_order : int
            Order above which the product is truncated.

        Returns
        -------
        Term
            Product of the two terms, or ``Term(coeff=0)`` if its order is
            above ``max_order`` or if one of the coefficients is zero.
        """
        if term1.order + term2.order > max_order:
            return Term(coeff=0)
        if _is_zero(term1.coeff) or _is_zero(term2.coeff):
            return Term(coeff=0)
        return Term(
            coeff=term1.coeff * term2.coeff,
            exp0=term1.exp0 + term2.exp0,
            exp1=term1.exp1 + term2.exp1,
            exp2=term1.exp2 + term2.exp2,
            exp3=term1.exp3 + term2.exp3,
        )

    @staticmethod
    def power_Term(term: Term, pow: int, max_order: int) -> Term:
        """
        A term raised to a given power, truncated at a given order.

        Parameters
        ----------
        term : Term
            Term to be raised to the power ``pow``.
        pow : int
            Power.
        max_order : int
            Order above which the result is truncated.

        Returns
        -------
        Term
            ``term ** pow``, or ``Term(coeff=0)`` if its order is above
            ``max_order`` or if the coefficient of ``term`` is zero.
        """
        if _is_zero(term.coeff):
            return Term(coeff=0)
        if term.order * pow > max_order:
            return Term(coeff=0)
        return Term(
            coeff=term.coeff**pow,
            exp0=term.exp0 * pow,
            exp1=term.exp1 * pow,
            exp2=term.exp2 * pow,
            exp3=term.exp3 * pow,
        )


# Internal "packed" representation of a term used in the hot loops:
# (exponents, order, coefficient).
_Packed = tuple[Exponents, int, Any]


def _pack_terms(terms: Iterable[Term]) -> list[_Packed]:
    """
    Convert terms to the internal packed representation, keeping their order.

    Parameters
    ----------
    terms : iterable of Term
        Terms to convert.

    Returns
    -------
    list of tuple
        ``(exponents, order, coeff)`` for every term.
    """
    return [
        (
            (t.exp0, t.exp1, t.exp2, t.exp3),
            t.exp0 + t.exp1 + t.exp2 + t.exp3,
            t.coeff,
        )
        for t in terms
    ]


def _pack_dict(coeffs: dict[Exponents, Any]) -> list[_Packed]:
    """
    Convert an ``{exponents: coeff}`` dictionary to the packed representation.

    Parameters
    ----------
    coeffs : dict
        Mapping from exponents to coefficients.

    Returns
    -------
    list of tuple
        ``(exponents, order, coeff)`` for every entry, in insertion order.
    """
    return [(e, e[0] + e[1] + e[2] + e[3], c) for e, c in coeffs.items()]


def _terms_from_dict(coeffs: dict[Exponents, Coeff]) -> list[Term]:
    """
    Build new :class:`Term` objects from an ``{exponents: coeff}`` dictionary.

    Parameters
    ----------
    coeffs : dict
        Mapping from exponents to coefficients.

    Returns
    -------
    list of Term
        One term per entry, in insertion order.
    """
    return [Term(c, e[0], e[1], e[2], e[3]) for e, c in coeffs.items()]


def _collect_nonzero(terms: Iterable[Term]) -> dict[Exponents, Coeff]:
    """
    Drop zero terms, then sum the coefficients of terms with equal exponents.

    Parameters
    ----------
    terms : iterable of Term
        Terms to collect.

    Returns
    -------
    dict
        Mapping from exponents to summed coefficients, in order of first
        appearance.
    """
    out: dict[Exponents, Coeff] = {}
    for t in terms:
        c = t.coeff
        if abs(c) <= _ATOL:
            continue
        key = (t.exp0, t.exp1, t.exp2, t.exp3)
        if key in out:
            out[key] = out[key] + c
        else:
            out[key] = c
    return out


def _mul_packed(
    terms1: list[_Packed], terms2: list[_Packed], max_order: int
) -> dict[Exponents, Coeff]:
    """
    Truncated product of two packed numerical polynomials.

    The result is the same as building all :meth:`Term.product_Terms` of
    the two term lists, removing zero terms and collecting equal monomials.

    Parameters
    ----------
    terms1 : list of tuple
        Packed terms of the first polynomial.
    terms2 : list of tuple
        Packed terms of the second polynomial.
    max_order : int
        Order above which the product is truncated.

    Returns
    -------
    dict
        Mapping from exponents to coefficients of the product.
    """
    atol = _ATOL
    rhs = [p for p in terms2 if not abs(p[2]) <= atol]
    out: dict[Exponents, Coeff] = {}
    for (a0, a1, a2, a3), oa, ca in terms1:
        if abs(ca) <= atol:
            continue
        limit = max_order - oa
        for eb, ob, cb in rhs:
            if ob > limit:
                continue
            c = ca * cb
            if abs(c) <= atol:
                continue
            key = (a0 + eb[0], a1 + eb[1], a2 + eb[2], a3 + eb[3])
            if key in out:
                out[key] = out[key] + c
            else:
                out[key] = c
    return out


@dataclass
class Polynom:
    """
    A polynomial of 4 variables with numerical coefficients.

    Parameters
    ----------
    terms : list of Term
        Terms of the polynomial.
    """

    terms: list[Term]

    def __str__(self) -> str:
        return "\n".join(map(str, self.terms))

    def __eq__(self, other: object) -> bool:
        """
        Compare two polynomials term by term, regardless of term order.

        Parameters
        ----------
        other : object
            Object to compare with.

        Returns
        -------
        bool
            ``True`` if both polynomials have the same number of terms and
            every term of one matches a distinct term of the other (see
            :meth:`Term.__eq__`). Comparison with an object that is not a
            :class:`Polynom` returns ``NotImplemented``.
        """
        if not isinstance(other, Polynom):
            return NotImplemented

        if len(self.terms) != len(other.terms):
            return False

        # Terms can only match if their exponents are equal, so group the
        # candidates by exponents instead of scanning all of them.
        candidates: dict[Exponents, list[Term]] = {}
        for t in other.terms:
            candidates.setdefault(t.exponents, []).append(t)

        for self_term in self.terms:
            group = candidates.get(self_term.exponents)
            if not group:
                return False
            for i, other_term in enumerate(group):
                if self_term == other_term:
                    del group[i]
                    break
            else:
                return False

        return True

    def truncate_at_order(self, max_order: int) -> None:
        """
        Remove, in place, all terms above a given order.

        Parameters
        ----------
        max_order : int
            Order above which terms are removed.
        """
        self.terms = [t for t in self.terms if t.order <= max_order]

    def remove_zero_terms(self) -> None:
        """
        Remove, in place, terms whose coefficient is zero within tolerance.
        """
        self.terms = [t for t in self.terms if not abs(t.coeff) <= _ATOL]

    def collect_terms(self) -> None:
        """
        Combine, in place, the coefficients of terms with equal exponents.

        The original :class:`Term` objects are not modified.
        """
        merged: dict[Exponents, Term] = {}
        for t in self.terms:
            key = t.exponents
            prev = merged.get(key)
            if prev is None:
                merged[key] = t
            else:
                merged[key] = Term(prev.coeff + t.coeff, *key)
        self.terms = list(merged.values())

    def substitute(
        self,
        val0: ArrayOrScalar,
        val1: ArrayOrScalar,
        val2: ArrayOrScalar,
        val3: ArrayOrScalar,
    ) -> ArrayOrScalar:
        """
        Evaluate the polynomial for given values of the four variables.

        Parameters
        ----------
        val0 : float, complex or numpy.ndarray
            Value(s) of the first variable.
        val1 : float, complex or numpy.ndarray
            Value(s) of the second variable.
        val2 : float, complex or numpy.ndarray
            Value(s) of the third variable.
        val3 : float, complex or numpy.ndarray
            Value(s) of the fourth variable.

        Returns
        -------
        complex or numpy.ndarray
            Value(s) of the polynomial. Arrays are broadcast together.
        """
        p0, p1, p2, p3 = _power_tables(
            (val0, val1, val2, val3), (t.exponents for t in self.terms)
        )
        total: Any = 0.0 + 0j
        for t in self.terms:
            total += t.coeff * p0[t.exp0] * p1[t.exp1] * p2[t.exp2] * p3[t.exp3]
        return total

    def get_max_order(self) -> int:
        """
        Maximum order of the polynomial.

        Returns
        -------
        int
            Highest total order among the terms, ``0`` for an empty
            polynomial.
        """
        return max((t.order for t in self.terms), default=0)

    def to_dict(self) -> dict[str, list]:
        """
        Representation of the polynomial with plain Python types.

        The result can be written to JSON. Floats are stored exactly.

        Returns
        -------
        dict
            ``{"exponents": [[e0, e1, e2, e3], ...], "coeff_real": [...],
            "coeff_imag": [...]}``, one entry per term, in term order.
        """
        coeffs = [complex(t.coeff) for t in self.terms]
        return {
            "exponents": [[int(e) for e in t.exponents] for t in self.terms],
            "coeff_real": [c.real for c in coeffs],
            "coeff_imag": [c.imag for c in coeffs],
        }

    @classmethod
    def from_dict(cls, data: dict[str, list]) -> Polynom:
        """
        Build a polynomial from the output of :meth:`to_dict`.

        Parameters
        ----------
        data : dict
            Dictionary with the keys ``"exponents"``, ``"coeff_real"`` and
            ``"coeff_imag"``.

        Returns
        -------
        Polynom
            Polynomial with complex coefficients.

        Raises
        ------
        ValueError
            If the dictionary is malformed.
        """
        try:
            exps, re, im = data["exponents"], data["coeff_real"], data["coeff_imag"]
        except (KeyError, TypeError) as err:
            raise ValueError(f"Invalid polynomial data: {err!r}.") from None
        if not len(exps) == len(re) == len(im):
            raise ValueError(
                "Invalid polynomial data: 'exponents', 'coeff_real' and "
                "'coeff_imag' must have the same length."
            )
        terms = []
        for e, r, i in zip(exps, re, im):
            if len(e) != NUM_VARS:
                raise ValueError(f"Invalid polynomial data: exponents {e}.")
            terms.append(Term(complex(r, i), *(int(v) for v in e)))
        return cls(terms=terms)

    @staticmethod
    def product_Coeff_Polynom(coeff: Coeff, poly: Polynom) -> Polynom:
        """
        Product of a numerical coefficient and a polynomial.

        Parameters
        ----------
        coeff : int, float or complex
            Coefficient to multiply with.
        poly : Polynom
            Polynomial to be multiplied.

        Returns
        -------
        Polynom
            New polynomial ``coeff * poly``.
        """
        return Polynom(
            terms=[
                Term(coeff * t.coeff, t.exp0, t.exp1, t.exp2, t.exp3)
                for t in poly.terms
            ]
        )

    @staticmethod
    def sum_Polynoms(poly1: Polynom, poly2: Polynom) -> Polynom:
        """
        Sum of two polynomials.

        Zero terms of the inputs are dropped and terms with equal exponents
        are collected.

        Parameters
        ----------
        poly1 : Polynom
            First polynomial in the sum.
        poly2 : Polynom
            Second polynomial in the sum.

        Returns
        -------
        Polynom
            New polynomial ``poly1 + poly2``, made of new :class:`Term`
            objects.
        """
        coeffs = _collect_nonzero(poly1.terms + poly2.terms)
        return Polynom(terms=_terms_from_dict(coeffs))

    @staticmethod
    def product_Polynoms(poly1: Polynom, poly2: Polynom, max_order: int) -> Polynom:
        """
        Product of two polynomials, truncated at a given order.

        Parameters
        ----------
        poly1 : Polynom
            First polynomial in the product.
        poly2 : Polynom
            Second polynomial in the product.
        max_order : int
            Order above which the product is truncated.

        Returns
        -------
        Polynom
            New polynomial ``poly1 * poly2`` without terms above
            ``max_order``.
        """
        coeffs = _mul_packed(_pack_terms(poly1.terms), _pack_terms(poly2.terms), max_order)
        return Polynom(terms=_terms_from_dict(coeffs))

    @staticmethod
    def power_Polynom(poly: Polynom, pow: int, max_order: int) -> Polynom:
        """
        A polynomial raised to a given power, truncated at a given order.

        Parameters
        ----------
        poly : Polynom
            Polynomial to be raised to the power ``pow``.
        pow : int
            Non-negative power.
        max_order : int
            Order above which the result is truncated.

        Returns
        -------
        Polynom
            ``poly ** pow``. For ``pow == 0`` this is the constant ``1``;
            for ``pow == 1`` the input object itself is returned.

        Raises
        ------
        ValueError
            If ``pow`` is negative.
        """
        _check_power(pow)
        if pow == 0:
            return Polynom(terms=[Term(coeff=1)])
        if pow == 1:
            return poly
        base = _pack_terms(poly.terms)
        result = _mul_packed(base, base, max_order)
        for _ in range(2, pow):
            result = _mul_packed(_pack_dict(result), base, max_order)
        return Polynom(terms=_terms_from_dict(result))

    @staticmethod
    def diff_Polynom(poly: Polynom, var: int) -> Polynom:
        """
        Derivative of a polynomial with respect to one variable.

        Parameters
        ----------
        poly : Polynom
            Polynomial to differentiate.
        var : int
            Index (``0``, ``1``, ``2`` or ``3``) of the variable with
            respect to which to differentiate.

        Returns
        -------
        Polynom
            New polynomial ``d poly / d v_var``.

        Raises
        ------
        ValueError
            If ``var`` is not a valid variable index.
        """
        _check_var(var)
        terms = []
        for t in poly.terms:
            exps = list(t.exponents)
            n = exps[var]
            if n > 0:
                exps[var] = n - 1
                terms.append(Term(t.coeff * n, *exps))
        return Polynom(terms=terms)


class _MapComposer:
    """
    Compose numerical polynomials with a fixed inner map.

    Powers of the inner polynomials and products of these powers are cached
    so that they are computed only once for all terms of all outer
    polynomials.

    Parameters
    ----------
    inner : sequence of 4 Polynom
        Components of the inner map.
    max_order : int
        Order above which all intermediate results are truncated.
    """

    def __init__(self, inner: Sequence[Polynom], max_order: int) -> None:
        self._inner = [_pack_terms(p.terms) for p in inner]
        self._max_order = max_order
        self._powers: list[dict[int, list[_Packed]]] = [{} for _ in range(NUM_VARS)]
        self._pairs: dict[tuple[int, int, int], list[_Packed]] = {}
        self._full: dict[Exponents, dict[Exponents, Coeff]] = {}

    def _power(self, var: int, k: int) -> list[_Packed]:
        """
        Packed ``inner[var] ** k`` (same as :meth:`Polynom.power_Polynom`).

        Parameters
        ----------
        var : int
            Index of the inner polynomial.
        k : int
            Power.

        Returns
        -------
        list of tuple
            Packed terms of the power.
        """
        cache = self._powers[var]
        if k not in cache:
            _check_power(k)
            if k == 0:
                cache[k] = [(_ZERO_EXPS, 0, 1)]
            elif k == 1:
                cache[k] = self._inner[var]
            else:
                base = self._inner[var]
                prev = base if k == 2 else self._power(var, k - 1)
                cache[k] = _pack_dict(_mul_packed(prev, base, self._max_order))
        return cache[k]

    def _pair(self, first: int, k1: int, k2: int) -> list[_Packed]:
        """
        Packed ``inner[first]**k1 * inner[first + 1]**k2``.

        Parameters
        ----------
        first : int
            Index of the first inner polynomial (``0`` or ``2``).
        k1 : int
            Power of the first inner polynomial.
        k2 : int
            Power of the second inner polynomial.

        Returns
        -------
        list of tuple
            Packed terms of the product.
        """
        key = (first, k1, k2)
        if key not in self._pairs:
            self._pairs[key] = _pack_dict(
                _mul_packed(
                    self._power(first, k1), self._power(first + 1, k2), self._max_order
                )
            )
        return self._pairs[key]

    def monomial(self, exps: Exponents) -> dict[Exponents, Coeff]:
        """
        The monomial ``prod_i inner[i] ** exps[i]``, truncated.

        Parameters
        ----------
        exps : tuple of 4 int
            Exponents of the outer monomial.

        Returns
        -------
        dict
            Mapping from exponents to coefficients.
        """
        if exps not in self._full:
            self._full[exps] = _mul_packed(
                self._pair(0, exps[0], exps[1]),
                self._pair(2, exps[2], exps[3]),
                self._max_order,
            )
        return self._full[exps]

    def compose(self, outer: Polynom) -> Polynom:
        """
        Substitute the inner map into an outer polynomial.

        Parameters
        ----------
        outer : Polynom
            Outer polynomial.

        Returns
        -------
        Polynom
            ``outer(inner)``, truncated at ``max_order``.
        """
        atol = _ATOL
        acc: dict[Exponents, Coeff] = {}
        for t in outer.terms:
            c = t.coeff
            touched = []
            for key, v in self.monomial(t.exponents).items():
                s = c * v
                if abs(s) <= atol:
                    continue
                if key in acc:
                    acc[key] = acc[key] + s
                    touched.append(key)
                else:
                    acc[key] = s
            # A coefficient that cancelled to zero is dropped (and is
            # re-inserted at the end if it appears again).
            for key in touched:
                if abs(acc[key]) <= atol:
                    del acc[key]
        return Polynom(terms=_terms_from_dict(acc))


@dataclass
class Map:
    """
    A polynomial map of 4 variables with numerical coefficients.

    Parameters
    ----------
    poly0 : Polynom
        Polynomial giving the first coordinate.
    poly1 : Polynom
        Polynomial giving the second coordinate.
    poly2 : Polynom
        Polynomial giving the third coordinate.
    poly3 : Polynom
        Polynomial giving the fourth coordinate.
    """

    poly0: Polynom
    poly1: Polynom
    poly2: Polynom
    poly3: Polynom

    @property
    def polys(self) -> tuple[Polynom, Polynom, Polynom, Polynom]:
        """
        The four component polynomials.

        Returns
        -------
        tuple of 4 Polynom
            ``(poly0, poly1, poly2, poly3)``.
        """
        return (self.poly0, self.poly1, self.poly2, self.poly3)

    def __str__(self) -> str:
        return "\n".join(f"poly{i}:\n{p}" for i, p in enumerate(self.polys))

    def __eq__(self, other: object) -> bool:
        """
        Compare two maps component by component.

        Parameters
        ----------
        other : object
            Object to compare with.

        Returns
        -------
        bool
            ``True`` if all four component polynomials are equal (see
            :meth:`Polynom.__eq__`). Comparison with an object that is not
            a :class:`Map` returns ``NotImplemented``.
        """
        if not isinstance(other, Map):
            return NotImplemented
        return all(p == q for p, q in zip(self.polys, other.polys))

    def substitute(
        self,
        val0: ArrayOrScalar,
        val1: ArrayOrScalar,
        val2: ArrayOrScalar,
        val3: ArrayOrScalar,
    ) -> tuple[ArrayOrScalar, ArrayOrScalar, ArrayOrScalar, ArrayOrScalar]:
        """
        Evaluate the map for given values of the four variables.

        Parameters
        ----------
        val0 : float, complex or numpy.ndarray
            Value(s) of the first variable.
        val1 : float, complex or numpy.ndarray
            Value(s) of the second variable.
        val2 : float, complex or numpy.ndarray
            Value(s) of the third variable.
        val3 : float, complex or numpy.ndarray
            Value(s) of the fourth variable.

        Returns
        -------
        tuple of 4 complex or numpy.ndarray
            Values of the four components of the map.
        """
        p0, p1, p2, p3 = _power_tables(
            (val0, val1, val2, val3),
            (t.exponents for poly in self.polys for t in poly.terms),
        )
        results = []
        for poly in self.polys:
            total: Any = 0.0 + 0j
            for t in poly.terms:
                total += t.coeff * p0[t.exp0] * p1[t.exp1] * p2[t.exp2] * p3[t.exp3]
            results.append(total)
        return (results[0], results[1], results[2], results[3])

    def get_max_order(self) -> int:
        """
        Maximum order of the map.

        Returns
        -------
        int
            Highest order among the four component polynomials.
        """
        return max(p.get_max_order() for p in self.polys)

    def to_dict(self) -> dict[str, dict[str, list]]:
        """
        Representation of the map with plain Python types.

        Returns
        -------
        dict
            ``{"poly0": ..., "poly1": ..., "poly2": ..., "poly3": ...}``
            with each component given by :meth:`Polynom.to_dict`.
        """
        return {f"poly{i}": p.to_dict() for i, p in enumerate(self.polys)}

    @classmethod
    def from_dict(cls, data: dict[str, dict[str, list]]) -> Map:
        """
        Build a map from the output of :meth:`to_dict`.

        Parameters
        ----------
        data : dict
            Dictionary with the keys ``"poly0"`` to ``"poly3"``.

        Returns
        -------
        Map
            The map.

        Raises
        ------
        ValueError
            If the dictionary is malformed.
        """
        try:
            polys = [data[f"poly{i}"] for i in range(NUM_VARS)]
        except (KeyError, TypeError) as err:
            raise ValueError(f"Invalid map data: {err!r}.") from None
        return cls(*(Polynom.from_dict(p) for p in polys))

    @staticmethod
    def composition_Map(map1: Map, map2: Map, max_order: int) -> Map:
        """
        Composition of two maps, truncated at a given order.

        Parameters
        ----------
        map1 : Map
            Inner map in the composition (applied first).
        map2 : Map
            Outer map in the composition (applied second).
        max_order : int
            Order above which the component polynomials are truncated.

        Returns
        -------
        Map
            The map ``map2(map1)``.
        """
        composer = _MapComposer(map1.polys, max_order)
        return Map(*(composer.compose(p) for p in map2.polys))
