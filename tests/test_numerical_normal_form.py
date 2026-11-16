from __future__ import annotations

import gzip
import json

import numpy as np
import pytest

from xnf4d.numerical_normal_form import (
    NormalForm4D,
    _first,
    _last,
    _monomial_exponents,
    _n_monomials,
    complex_to_real,
    real_to_complex,
)
from xnf4d.polynom import Map, Polynom, Term

# --------------------------------------------------------------------------- #
# Test maps
# --------------------------------------------------------------------------- #

QX, QY = 0.2111, 0.1532


def P(*terms) -> Polynom:
    return Polynom([Term(c, *e) for c, e in terms])


def identity_map() -> Map:
    return Map(
        P((1.0, (1, 0, 0, 0))),
        P((1.0, (0, 1, 0, 0))),
        P((1.0, (0, 0, 1, 0))),
        P((1.0, (0, 0, 0, 1))),
    )


def henon_real(qx: float, qy: float, k2: float, k3: float) -> Map:
    """
    Real symplectic map: kick from the potential
    V = k2/3 (x^3 - 3xy^2) + k3/4 (x^4 - 6x^2y^2 + y^4), then a rotation.
    The map is exactly a polynomial of order 3.
    """
    kick = Map(
        P((1.0, (1, 0, 0, 0))),
        P(
            (1.0, (0, 1, 0, 0)),
            (-k2, (2, 0, 0, 0)),
            (k2, (0, 0, 2, 0)),
            (-k3, (3, 0, 0, 0)),
            (3 * k3, (1, 0, 2, 0)),
        ),
        P((1.0, (0, 0, 1, 0))),
        P(
            (1.0, (0, 0, 0, 1)),
            (2 * k2, (1, 0, 1, 0)),
            (3 * k3, (2, 0, 1, 0)),
            (-k3, (0, 0, 3, 0)),
        ),
    )
    cx, sx = np.cos(2 * np.pi * qx), np.sin(2 * np.pi * qx)
    cy, sy = np.cos(2 * np.pi * qy), np.sin(2 * np.pi * qy)
    rot = Map(
        P((cx, (1, 0, 0, 0)), (sx, (0, 1, 0, 0))),
        P((-sx, (1, 0, 0, 0)), (cx, (0, 1, 0, 0))),
        P((cy, (0, 0, 1, 0)), (sy, (0, 0, 0, 1))),
        P((-sy, (0, 0, 1, 0)), (cy, (0, 0, 0, 1))),
    )
    return Map.composition_Map(kick, rot, 3)


def two_kick_real(qx: float, qy: float) -> Map:
    """Exact composition of two different kicks (order 9, no symmetry)."""
    first = henon_real(0.6 * qx, 0.6 * qy, 0.7, 0.4)
    second = henon_real(0.4 * qx, 0.4 * qy, -0.3, 0.9)
    return Map.composition_Map(first, second, 9)


def real_map_to_complex(real_map: Map, order: int) -> Map:
    """L o M o L^-1 with z = x - i px and z* = x + i px."""
    to_real = Map(
        P((0.5, (1, 0, 0, 0)), (0.5, (0, 1, 0, 0))),
        P((0.5j, (1, 0, 0, 0)), (-0.5j, (0, 1, 0, 0))),
        P((0.5, (0, 0, 1, 0)), (0.5, (0, 0, 0, 1))),
        P((0.5j, (0, 0, 1, 0)), (-0.5j, (0, 0, 0, 1))),
    )
    to_complex = Map(
        P((1.0, (1, 0, 0, 0)), (-1j, (0, 1, 0, 0))),
        P((1.0, (1, 0, 0, 0)), (1j, (0, 1, 0, 0))),
        P((1.0, (0, 0, 1, 0)), (-1j, (0, 0, 0, 1))),
        P((1.0, (0, 0, 1, 0)), (1j, (0, 0, 0, 1))),
    )
    inner = Map.composition_Map(to_real, real_map, order)
    return Map.composition_Map(inner, to_complex, order)


def eigs(qx: float, qy: float) -> list[complex]:
    return [
        np.exp(2j * np.pi * qx),
        np.exp(-2j * np.pi * qx),
        np.exp(2j * np.pi * qy),
        np.exp(-2j * np.pi * qy),
    ]


# --------------------------------------------------------------------------- #
# Polynomial helpers
# --------------------------------------------------------------------------- #


def as_dict(poly: Polynom) -> dict:
    out: dict = {}
    for t in poly.terms:
        out[t.exponents] = out.get(t.exponents, 0) + t.coeff
    return out


def truncated(poly: Polynom, order: int) -> Polynom:
    p = Polynom(list(poly.terms))
    p.truncate_at_order(order)
    return p


def max_abs_coeff(poly: Polynom) -> float:
    return max((abs(t.coeff) for t in poly.terms), default=0.0)


def assert_poly_close(p: Polynom, q: Polynom, atol: float, err_msg: str = "") -> None:
    a, b = as_dict(p), as_dict(q)
    for key in set(a) | set(b):
        diff = abs(a.get(key, 0) - b.get(key, 0))
        assert diff <= atol, f"{err_msg} {key}: {a.get(key, 0)} vs {b.get(key, 0)}"


def poisson(f: Polynom, g: Polynom, order: int) -> Polynom:
    """{f, g} = sum over pairs (0, 1), (2, 3) of df/da dg/db - df/db dg/da."""
    total = Polynom([])
    for a, b in ((0, 1), (2, 3)):
        fa, fb = Polynom.diff_Polynom(f, a), Polynom.diff_Polynom(f, b)
        ga, gb = Polynom.diff_Polynom(g, a), Polynom.diff_Polynom(g, b)
        total = Polynom.sum_Polynoms(total, Polynom.product_Polynoms(fa, gb, order))
        total = Polynom.sum_Polynoms(
            total,
            Polynom.product_Coeff_Polynom(-1.0, Polynom.product_Polynoms(fb, ga, order)),
        )
    return total


def assert_symplectic(m: Map, order: int, atol: float) -> None:
    """Poisson brackets of the components are canonical up to ``order``."""
    J = {(0, 1): 1.0, (2, 3): 1.0}
    for a in range(4):
        for b in range(a + 1, 4):
            br = poisson(m.polys[a], m.polys[b], order)
            br.truncate_at_order(order)
            expected = Polynom([Term(J.get((a, b), 0.0))])
            assert_poly_close(br, expected, atol, err_msg=f"{{{a}, {b}}}")


def is_resonant(k1: int, k2: int, bases: list[tuple[int, int]]) -> bool:
    """Whether (k1, k2) is an integer combination of the resonance vectors."""
    if k1 == 0 and k2 == 0:
        return True
    if len(bases) == 1:
        n, m = bases[0]
        return k1 * m - k2 * n == 0 and (k1 * n + k2 * m) % (n * n + m * m) == 0
    if len(bases) == 2:
        (n1, m1), (n2, m2) = bases
        det = n1 * m2 - n2 * m1
        return (k1 * m2 - k2 * n2) % det == 0 and (n1 * k2 - m1 * k1) % det == 0
    return False


_SHIFT = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def rk4_flow(field: Map, z, steps: int = 60):
    h = 1.0 / steps
    z = list(z)
    for _ in range(steps):
        k1 = field.substitute(*z)
        k2 = field.substitute(*[a + h / 2 * b for a, b in zip(z, k1)])
        k3 = field.substitute(*[a + h / 2 * b for a, b in zip(z, k2)])
        k4 = field.substitute(*[a + h * b for a, b in zip(z, k3)])
        z = [
            a + h / 6 * (b1 + 2 * b2 + 2 * b3 + b4)
            for a, b1, b2, b3, b4 in zip(z, k1, k2, k3, k4)
        ]
    return z


def physical_point(rng, amp: float):
    w = amp * (rng.normal(size=2) + 1j * rng.normal(size=2))
    return [w[0], np.conj(w[0]), w[1], np.conj(w[1])]


# --------------------------------------------------------------------------- #
# Normal form cases (computed once per test session)
# --------------------------------------------------------------------------- #

CASES = {
    # name: (real map factory, map order, nf order, res_case, res_eig, bases)
    "nonresonant": (lambda: henon_real(QX, QY, 0.7, 0.4), 3, 6, "nonresonant", None, []),
    "nonresonant_two_kick": (
        lambda: two_kick_real(QX, QY), 9, 5, "nonresonant", None, []
    ),
    "resonant_exact_tune": (
        lambda: two_kick_real(1 / 3, QY), 9, 5, "resonant", None, [(3, 0)]
    ),
    "resonant_near_tune": (
        lambda: two_kick_real(0.3345, QY), 9, 5, "resonant", None, [(3, 0)]
    ),
    "quasiresonant_near_tune": (
        lambda: two_kick_real(0.3345, QY),
        9,
        5,
        "quasiresonant",
        eigs(1 / 3, QY),
        [(3, 0)],
    ),
    "double_resonance": (
        lambda: two_kick_real(1 / 3, 1 / 6), 9, 4, "resonant", None, [(3, 0), (1, -2)]
    ),
}


class Case:
    def __init__(self, name: str) -> None:
        factory, f_order, nf_order, rc, res_eig, bases = CASES[name]
        self.name = name
        self.real_map = factory()
        self.F = real_map_to_complex(self.real_map, f_order)
        self.nf_order = nf_order
        self.res_case = rc
        self.bases = bases
        # Relative accuracy: the quasi-resonant second step works with the
        # small detuning as linear tunes, which amplifies rounding errors.
        self.rtol = 1e-8 if rc == "quasiresonant" else 1e-12
        kwargs = {}
        if res_eig is not None:
            kwargs["res_eig"] = res_eig
        if bases:
            kwargs["res_basis1"] = list(bases[0])
        if len(bases) > 1:
            kwargs["res_basis2"] = list(bases[1])
        self.nf = NormalForm4D(self.F, nf_order, rc, **kwargs)
        self.nf.compute_normal_form()
        self.N = Map.composition_Map(
            Map.composition_Map(self.nf.Phi, self.F, nf_order), self.nf.Psi, nf_order
        )


_case_cache: dict[str, Case] = {}


@pytest.fixture(params=list(CASES))
def case(request) -> Case:
    if request.param not in _case_cache:
        _case_cache[request.param] = Case(request.param)
    return _case_cache[request.param]


def get_case(name: str) -> Case:
    if name not in _case_cache:
        _case_cache[name] = Case(name)
    return _case_cache[name]


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(2024)


# --------------------------------------------------------------------------- #
# Coordinate helpers
# --------------------------------------------------------------------------- #


class TestCoordinates:
    def test_real_to_complex(self):
        z1, z1s, z2, z2s = real_to_complex(1.0, 2.0, 3.0, 4.0)
        assert (z1, z1s, z2, z2s) == (1 - 2j, 1 + 2j, 3 - 4j, 3 + 4j)

    def test_roundtrip(self, rng):
        x = rng.normal(size=(4, 10))
        back = complex_to_real(*real_to_complex(*x))
        for a, b in zip(back, x):
            np.testing.assert_allclose(a, b, atol=1e-15)
            assert np.iscomplexobj(a)

    def test_complex_to_real_general(self):
        # Also defined for non-conjugate (complex) inputs.
        x, px, y, py = complex_to_real(1 + 1j, 2.0, 3.0, 1j)
        assert real_to_complex(x, px, y, py) == pytest.approx((1 + 1j, 2.0, 3.0, 1j))


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #


class TestIndexing:
    def test_monomial_counts(self):
        assert [_n_monomials(o) for o in range(-1, 4)] == [0, 1, 5, 15, 35]
        assert _first(0) == 0 and _last(0) == 0
        assert _first(2) == 5 and _last(2) == 14

    def test_ordering(self):
        exps = _monomial_exponents(3)
        assert exps[:5].tolist() == [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
        ]
        assert exps[_first(3)].tolist() == [0, 0, 0, 3]
        assert exps[_last(3)].tolist() == [3, 0, 0, 0]
        orders = exps.sum(axis=1)
        assert np.all(np.diff(orders) >= 0)

    def test_linear_eval_factors(self, rng):
        vec = list(np.exp(2j * np.pi * rng.random(4)))
        f = NormalForm4D._lin_eval_factors(vec, 4)
        exps = _monomial_exponents(4)
        direct = np.prod(np.array(vec)[None, :] ** exps, axis=1)
        np.testing.assert_allclose(f, direct, rtol=1e-13)
        g = NormalForm4D._hom_eq_factors(vec, 3)
        np.testing.assert_allclose(g, direct[_first(3) : _last(3) + 1], rtol=1e-13)

    def test_fmod(self):
        assert NormalForm4D.fmod(7, 3) == 1
        assert NormalForm4D.fmod(-7, 3) == -1
        assert NormalForm4D.fmod(7, -3) == 1
        assert NormalForm4D.fmod(6, 3) == 0

    def test_lie_derivative_matches_polynom_algebra(self, rng):
        nf = NormalForm4D(identity_map(), 5)
        exps = _monomial_exponents(5)
        A = np.zeros((4, nf.N_dim), dtype=complex)
        C = np.zeros((4, nf.N_dim), dtype=complex)
        nA, nC = _n_monomials(3), _n_monomials(4)
        A[:, :nA] = rng.normal(size=(4, nA)) + 1j * rng.normal(size=(4, nA))
        C[:, :nC] = rng.normal(size=(4, nC)) + 1j * rng.normal(size=(4, nC))
        R = np.zeros_like(A)
        R = nf._lie_derivative(A, C, R, 3, 0, 4, 0, 5, (3, 4, 5))

        def poly(row, n):
            return Polynom([Term(c, *e) for c, e in zip(row[:n], exps[:n].tolist())])

        A_p = [poly(A[k], nA) for k in range(4)]
        for j in range(4):
            Cj = poly(C[j], nC)
            expected = Polynom([])
            for k in range(4):
                expected = Polynom.sum_Polynoms(
                    expected,
                    Polynom.product_Polynoms(A_p[k], Polynom.diff_Polynom(Cj, k), 5),
                )
            assert_poly_close(poly(R[j], _n_monomials(5)), expected, atol=1e-12)


# --------------------------------------------------------------------------- #
# Construction and validation
# --------------------------------------------------------------------------- #


class TestValidation:
    def test_default_is_nonresonant(self):
        nf = NormalForm4D(identity_map(), 3)
        assert nf.res_case == "nonresonant"
        assert nf.res_space_dim == 0

    def test_res_space_dim_from_bases(self):
        nf = NormalForm4D(identity_map(), 3, "resonant", res_basis1=[3, 0])
        assert nf.res_space_dim == 1
        nf = NormalForm4D(
            identity_map(), 3, "resonant", res_basis1=[3, 0], res_basis2=[1, -2]
        )
        assert nf.res_space_dim == 2

    @pytest.mark.parametrize("res_case", ["exact", "", 0, 1, 2, None])
    def test_invalid_res_case(self, res_case):
        with pytest.raises(ValueError, match="res_case"):
            NormalForm4D(identity_map(), 3, res_case)

    def test_quasiresonant_needs_res_eig(self):
        with pytest.raises(ValueError, match="res_eig"):
            NormalForm4D(identity_map(), 3, "quasiresonant", res_basis1=[3, 0])
        with pytest.raises(ValueError, match="res_eig"):
            NormalForm4D(
                identity_map(), 3, "quasiresonant", res_eig=[1, 1], res_basis1=[3, 0]
            )

    @pytest.mark.parametrize("res_case", ["nonresonant", "resonant"])
    def test_res_eig_only_for_quasiresonant(self, res_case):
        kwargs = {"res_basis1": [3, 0]} if res_case == "resonant" else {}
        with pytest.raises(ValueError, match="res_eig"):
            NormalForm4D(identity_map(), 3, res_case, res_eig=eigs(QX, QY), **kwargs)

    @pytest.mark.parametrize("res_case", ["resonant", "quasiresonant"])
    def test_resonant_needs_basis(self, res_case):
        kwargs = {"res_eig": eigs(QX, QY)} if res_case == "quasiresonant" else {}
        with pytest.raises(ValueError, match="res_basis1"):
            NormalForm4D(identity_map(), 3, res_case, **kwargs)

    def test_nonresonant_rejects_basis(self):
        with pytest.raises(ValueError, match="non-resonant"):
            NormalForm4D(identity_map(), 3, res_basis1=[3, 0])

    def test_second_basis_needs_first(self):
        with pytest.raises(ValueError, match="res_basis2"):
            NormalForm4D(identity_map(), 3, "resonant", res_basis2=[3, 0])

    @pytest.mark.parametrize("basis", [[0, 0], [3], [1, 2, 3]])
    def test_invalid_basis(self, basis):
        with pytest.raises(ValueError, match="resonance vector"):
            NormalForm4D(identity_map(), 3, "resonant", res_basis1=basis)

    def test_dependent_bases(self):
        with pytest.raises(ValueError, match="linearly independent"):
            NormalForm4D(
                identity_map(), 3, "resonant", res_basis1=[3, 0], res_basis2=[6, 0]
            )

    def test_invalid_order(self):
        with pytest.raises(ValueError, match="max_nf_order"):
            NormalForm4D(identity_map(), 0)

    def test_negative_exponent(self):
        m = identity_map()
        m.poly0 = Polynom(m.poly0.terms + [Term(1.0, -1, 2, 0, 0)])
        with pytest.raises(ValueError, match="negative"):
            NormalForm4D(m, 3)

    def test_terms_above_nf_order_are_ignored(self):
        F = real_map_to_complex(two_kick_real(QX, QY), 9)
        Ft = Map(*(truncated(p, 4) for p in F.polys))
        a, b = NormalForm4D(F, 4), NormalForm4D(Ft, 4)
        a.compute_normal_form()
        b.compute_normal_form()
        for x, y in ((a.Phi, b.Phi), (a.Psi, b.Psi), (a.U, b.U), (a.N, b.N)):
            for p, q in zip(x.polys, y.polys):
                assert as_dict(p) == as_dict(q)
        assert as_dict(a.H) == as_dict(b.H)

    def test_attributes(self):
        nf = NormalForm4D(real_map_to_complex(henon_real(QX, QY, 0.7, 0.4), 3), 5)
        assert nf.nf_order == 5
        assert nf.nord == 6
        assert nf.N_dim == _n_monomials(6)
        assert nf.N_dim_H == _n_monomials(7)
        np.testing.assert_allclose(nf.lin_eig, eigs(QX, QY), atol=1e-14)
        assert all(
            getattr(nf, name) is None for name in ("Phi", "Psi", "N", "U", "H")
        )

    def test_resonance_bases_storage(self):
        nf = NormalForm4D(
            identity_map(), 3, "resonant", res_basis1=[3, 0], res_basis2=[1, -2]
        )
        # Stored "transposed": [n of both resonances], [m of both resonances]
        assert nf.res_basis1 == [3, 1]
        assert nf.res_basis2 == [0, -2]

    def test_quasiresonant_keeps_attributes(self):
        c = get_case("quasiresonant_near_tune")
        np.testing.assert_allclose(c.nf.lin_eig, eigs(0.3345, QY), atol=1e-14)
        assert c.nf.res_basis1 == [3, 0] and c.nf.res_basis2 == [0, 0]

    def test_duplicate_terms_are_summed(self):
        F = real_map_to_complex(henon_real(QX, QY, 0.7, 0.4), 3)
        split = Map(
            *(
                Polynom(
                    [Term(t.coeff / 2, *t.exponents) for t in p.terms]
                    + [Term(t.coeff / 2, *t.exponents) for t in p.terms]
                )
                for p in F.polys
            )
        )
        a = NormalForm4D(F, 4)
        b = NormalForm4D(split, 4)
        a.compute_normal_form()
        b.compute_normal_form()
        assert a.H == b.H


# --------------------------------------------------------------------------- #
# Simple cases with known results
# --------------------------------------------------------------------------- #


class TestLinearMap:
    @pytest.fixture
    def nf(self):
        lam = eigs(QX, QY)
        F = Map(*(P((lam[j], tuple(int(i == j) for i in range(4)))) for j in range(4)))
        nf = NormalForm4D(F, 4)
        nf.compute_normal_form()
        return nf

    def test_transformations_are_identity(self, nf):
        assert nf.Phi == identity_map()
        assert nf.Psi == identity_map()

    def test_vector_field_and_hamiltonian(self, nf):
        lam = eigs(QX, QY)
        for j in range(4):
            e = tuple(int(i == j) for i in range(4))
            assert nf.U.polys[j] == P((np.log(lam[j]), e))
        assert nf.H == P(
            (2j * np.pi * QX, (1, 1, 0, 0)), (2j * np.pi * QY, (0, 0, 1, 1))
        )


class TestReferenceValues:
    """Coefficients computed with the original implementation."""

    def test_nonresonant_henon(self):
        F = real_map_to_complex(henon_real(QX, QY, 0.7, 0.4), 3)
        nf = NormalForm4D(F, 4)
        nf.compute_normal_form()
        h = as_dict(nf.H)
        expected_h = {
            (1, 1, 0, 0): 1.3263804183456107j,
            (0, 0, 1, 1): 0.9625839890599126j,
            (2, 2, 0, 0): 0.02301409353960188j,
            (1, 1, 1, 1): -0.3382556799727904j,
            (0, 0, 2, 2): 0.04704828278254417j,
        }
        for k, v in expected_h.items():
            assert h[k] == pytest.approx(v, abs=1e-14)
        phi0 = as_dict(nf.Phi.poly0)
        expected_phi = {
            (2, 0, 0, 0): 0.11200295125478617 - 0.08749999999999997j,
            (0, 2, 0, 0): 0.03894653113351202 - 0.0875j,
            (0, 0, 1, 1): 0.22400590250957228 + 0.17499999999999993j,
            (2, 1, 0, 0): -0.011027828802448375,
        }
        for k, v in expected_phi.items():
            assert phi0[k] == pytest.approx(v, abs=1e-14)


# --------------------------------------------------------------------------- #
# Properties of the normal form, for all cases
# --------------------------------------------------------------------------- #


class TestNormalFormProperties:
    def test_input_map_is_symplectic(self, case):
        assert_symplectic(case.F, case.nf_order, atol=1e-12)

    def test_phi_symplectic(self, case):
        assert_symplectic(case.nf.Phi, case.nf_order - 1, atol=1e-11)

    def test_psi_symplectic(self, case):
        assert_symplectic(case.nf.Psi, case.nf_order - 1, atol=1e-11)

    def test_psi_inverse_of_phi(self, case):
        n = case.nf_order
        for comp in (
            Map.composition_Map(case.nf.Phi, case.nf.Psi, n),
            Map.composition_Map(case.nf.Psi, case.nf.Phi, n),
        ):
            for got, want in zip(comp.polys, identity_map().polys):
                assert_poly_close(got, want, atol=1e-11)

    def test_linear_part_is_identity(self, case):
        for m in (case.nf.Phi, case.nf.Psi):
            for got, want in zip(m.polys, identity_map().polys):
                assert_poly_close(truncated(got, 1), want, atol=1e-15)

    def test_orders(self, case):
        for m in (case.nf.Phi, case.nf.Psi, case.nf.U):
            assert m.get_max_order() <= case.nf_order
        assert case.nf.H.get_max_order() <= case.nf_order + 1

    def test_normal_form_map_attribute(self, case):
        # N equals Psi o F o Phi computed with the polynomial algebra
        scale = max(max_abs_coeff(p) for p in case.N.polys)
        for got, want in zip(case.nf.N.polys, case.N.polys):
            assert_poly_close(got, want, atol=1e-11 * scale)
        lam = case.nf.lin_eig
        for j, p in enumerate(case.nf.N.polys):
            e = tuple(int(i == j) for i in range(4))
            assert as_dict(truncated(p, 1)) == pytest.approx({e: lam[j]}, abs=1e-14)
        assert case.nf.N.get_max_order() <= case.nf_order

    def test_normal_form_has_only_resonant_terms(self, case):
        scale = max(max_abs_coeff(p) for p in case.nf.N.polys)
        for j, poly in enumerate(case.nf.N.polys):
            for t in poly.terms:
                n1, n2, n3, n4 = t.exponents
                k1, k2 = n1 - n2 - _SHIFT[j][0], n3 - n4 - _SHIFT[j][1]
                if not is_resonant(k1, k2, case.bases):
                    assert abs(t.coeff) < 1e-9 * scale, (j, t)

    def test_vector_field_has_only_resonant_terms(self, case):
        for j, poly in enumerate(case.nf.U.polys):
            for t in poly.terms:
                n1, n2, n3, n4 = t.exponents
                k1, k2 = n1 - n2 - _SHIFT[j][0], n3 - n4 - _SHIFT[j][1]
                assert is_resonant(k1, k2, case.bases), (j, t)

    def test_hamiltonian_generates_vector_field(self, case):
        H, U = case.nf.H, case.nf.U
        derivs = [
            Polynom.diff_Polynom(H, 1),
            Polynom.product_Coeff_Polynom(-1.0, Polynom.diff_Polynom(H, 0)),
            Polynom.diff_Polynom(H, 3),
            Polynom.product_Coeff_Polynom(-1.0, Polynom.diff_Polynom(H, 2)),
        ]
        tol = case.rtol * max(max_abs_coeff(p) for p in U.polys)
        for j in range(4):
            assert_poly_close(
                truncated(derivs[j], case.nf_order), U.polys[j], atol=tol, err_msg=f"U{j}"
            )

    def test_hamiltonian_reality(self, case):
        # H = i * (real-valued function on physical points):
        # h_(a, b, c, d) = -conj(h_(b, a, d, c))
        h = as_dict(case.nf.H)
        scale = max(abs(v) for v in h.values())
        for (a, b, c, d), v in h.items():
            partner = h.get((b, a, d, c), 0)
            assert abs(v + np.conj(partner)) < 1e-8 * scale, (a, b, c, d)

    def test_hamiltonian_linear_part(self, case):
        h = as_dict(case.nf.H)
        lam = np.array(case.nf.lin_eig)
        if case.res_case == "quasiresonant":
            # U and H describe the map in the frame rotating with res_eig
            lam = lam / np.array(case.nf.res_eig)
        assert h.get((1, 1, 0, 0), 0) == pytest.approx(np.log(lam[0]), abs=1e-12)
        assert h.get((0, 0, 1, 1), 0) == pytest.approx(np.log(lam[2]), abs=1e-12)

    @pytest.mark.parametrize("name", ["Phi", "Psi", "U"])
    def test_maps_preserve_reality(self, case, name):
        # Component 1 is the complex conjugate of component 0 with the
        # exponents of conjugate variables swapped (same for 2 and 3).
        m = getattr(case.nf, name)
        for j0, j1 in ((0, 1), (2, 3)):
            a, b = as_dict(m.polys[j0]), as_dict(m.polys[j1])
            scale = max([abs(v) for v in a.values()] + [1e-300])
            for (n1, n2, n3, n4), v in a.items():
                partner = b.get((n2, n1, n4, n3), 0)
                assert abs(v - np.conj(partner)) < 1e-8 * scale

    def test_flow_of_vector_field_gives_normal_form(self, case, rng):
        """
        res_case 0/1: N = Lambda o flow(U_nonlinear);
        res_case 2:   N = Lambda_res o flow(U).
        The error must scale like amp ** (nf_order + 1).
        """
        nf = case.nf
        if case.res_case == "quasiresonant":
            field, lam = nf.U, nf.res_eig
        else:
            field = Map(*(Polynom([t for t in p.terms if t.order != 1]) for p in nf.U.polys))
            lam = nf.lin_eig
        w = physical_point(rng, 1.0)
        errors = []
        for amp in (2e-2, 1e-2):
            z = [amp * v for v in w]
            N = nf.Psi.substitute(*case.F.substitute(*nf.Phi.substitute(*z)))
            pred = [l * v for l, v in zip(lam, rk4_flow(field, z))]
            errors.append(max(abs(a - b) for a, b in zip(N, pred)))
        assert errors[1] < 1e-12 or errors[0] / errors[1] > 0.7 * 2 ** (case.nf_order + 1)

    def test_recompute_gives_same_result(self, case):
        before = (case.nf.Phi, case.nf.Psi, case.nf.N, case.nf.U, case.nf.H)
        case.nf.compute_normal_form()
        after = (case.nf.Phi, case.nf.Psi, case.nf.N, case.nf.U, case.nf.H)
        assert before == after


class TestNonresonant:
    def test_hamiltonian_is_imaginary_and_depends_on_actions_only(self):
        c = get_case("nonresonant")
        for t in c.nf.H.terms:
            n1, n2, n3, n4 = t.exponents
            assert n1 == n2 and n3 == n4
            assert abs(t.coeff.real) < 1e-12

    def test_full_flow_of_vector_field(self, rng):
        # In the non-resonant case U (with its linear part) interpolates N.
        c = get_case("nonresonant_two_kick")
        w = physical_point(rng, 1e-2)
        N = c.nf.Psi.substitute(*c.F.substitute(*c.nf.Phi.substitute(*w)))
        flow = rk4_flow(c.nf.U, w, steps=200)
        for a, b in zip(N, flow):
            assert abs(a - b) < 1e-10

    def test_resonant_quantities_in_resonant_case_are_not_imaginary(self):
        # The real parts of H only vanish in the non-resonant case.
        c = get_case("resonant_exact_tune")
        assert max(abs(t.coeff.real) for t in c.nf.H.terms) > 1e-3

    def test_tracking_actions_are_invariant(self, rng):
        c = get_case("nonresonant")
        x = np.array([0.02, -0.01, 0.015, 0.005])
        actions, cs_actions = [], []
        for _ in range(50):
            z = real_to_complex(*x)
            zeta = c.nf.norm_to_nf(*z)
            actions.append([abs(zeta[0]) ** 2, abs(zeta[2]) ** 2])
            cs_actions.append([abs(z[0]) ** 2, abs(z[2]) ** 2])
            x = np.array([v.real for v in c.real_map.substitute(*x)])

        def spread(a):
            a = np.array(a)
            return (a.max(axis=0) - a.min(axis=0)) / a.mean(axis=0)

        assert np.all(spread(actions) < 1e-6)
        # in Courant-Snyder coordinates the "actions" vary much more
        assert np.all(spread(cs_actions) > 1e3 * spread(actions))


class TestTransformations:
    def test_not_computed(self):
        nf = NormalForm4D(identity_map(), 3)
        with pytest.raises(ValueError):
            nf.norm_to_nf(0.1, 0.1, 0.1, 0.1)
        with pytest.raises(ValueError):
            nf.nf_to_norm(0.1, 0.1, 0.1, 0.1)

    def test_roundtrip_and_complex_output(self, rng):
        c = get_case("nonresonant")
        z = [rng.normal(size=5) * 1e-2 for _ in range(4)]
        zc = real_to_complex(*z)
        zeta = c.nf.norm_to_nf(*zc)
        back = c.nf.nf_to_norm(*zeta)
        for a, b in zip(back, zc):
            assert np.iscomplexobj(a)
            np.testing.assert_allclose(a, b, atol=1e-11)
        for a, b in zip(complex_to_real(*back), z):
            np.testing.assert_allclose(a.real, b, atol=1e-11)
            np.testing.assert_allclose(a.imag, 0, atol=1e-11)

    def test_physical_points_stay_physical(self, rng):
        c = get_case("resonant_exact_tune")
        zc = real_to_complex(*(rng.normal(size=(4, 6)) * 0.05))
        zeta = c.nf.norm_to_nf(*zc)
        np.testing.assert_allclose(zeta[1], np.conj(zeta[0]), atol=1e-14)
        np.testing.assert_allclose(zeta[3], np.conj(zeta[2]), atol=1e-14)

    def test_match_maps(self, rng):
        c = get_case("nonresonant")
        z = [0.01 + 0.02j, 0.01 - 0.02j, -0.03j, 0.03j]
        assert c.nf.norm_to_nf(*z) == c.nf.Psi.substitute(*z)
        assert c.nf.nf_to_norm(*z) == c.nf.Phi.substitute(*z)


class TestSmallDenominators:
    def test_exact_undeclared_resonance_raises(self):
        # Qx = Qy = 1/4 exactly with an octupole-like term z1*^3: the
        # 4 Qx resonance gives an exactly vanishing denominator.
        F = Map(
            P((1j, (1, 0, 0, 0)), (0.1, (0, 3, 0, 0))),
            P((-1j, (0, 1, 0, 0)), (0.1, (3, 0, 0, 0))),
            P((1j, (0, 0, 1, 0))),
            P((-1j, (0, 0, 0, 1))),
        )
        nf = NormalForm4D(F, 4)
        with pytest.raises(ZeroDivisionError):
            nf.compute_normal_form()


# --------------------------------------------------------------------------- #
# Saving and loading
# --------------------------------------------------------------------------- #


def assert_same_results(a: NormalForm4D, b: NormalForm4D) -> None:
    """Results are identical (term order, exponents and coefficients)."""
    for name in ("Phi", "Psi", "N", "U"):
        for p, q in zip(getattr(a, name).polys, getattr(b, name).polys):
            assert [(t.exponents, t.coeff) for t in p.terms] == [
                (t.exponents, t.coeff) for t in q.terms
            ], name
    assert [(t.exponents, t.coeff) for t in a.H.terms] == [
        (t.exponents, t.coeff) for t in b.H.terms
    ]


class TestSaveLoad:
    @pytest.mark.parametrize("suffix", [".json", ".json.gz"])
    def test_roundtrip_is_exact(self, case, tmp_path, suffix):
        path = tmp_path / f"nf{suffix}"
        case.nf.to_json(path)
        loaded = NormalForm4D.from_json(path)
        assert_same_results(case.nf, loaded)
        assert loaded.res_case == case.nf.res_case
        assert loaded.res_space_dim == case.nf.res_space_dim
        assert loaded.res_basis1 == case.nf.res_basis1
        assert loaded.res_basis2 == case.nf.res_basis2
        assert loaded.res_eig == case.nf.res_eig
        assert loaded.nf_order == case.nf.nf_order
        assert loaded.lin_eig == case.nf.lin_eig

    def test_gzip_is_compressed_and_detected(self, tmp_path):
        nf = get_case("nonresonant").nf
        nf.to_json(tmp_path / "nf.json.gz")
        nf.to_json(tmp_path / "nf.json")
        assert (tmp_path / "nf.json.gz").read_bytes()[:2] == b"\x1f\x8b"
        assert (tmp_path / "nf.json.gz").stat().st_size < (tmp_path / "nf.json").stat().st_size
        # detection does not rely on the file name
        (tmp_path / "nf.json.gz").rename(tmp_path / "renamed.json")
        assert_same_results(nf, NormalForm4D.from_json(tmp_path / "renamed.json"))

    def test_string_path_and_indent(self, tmp_path):
        nf = get_case("nonresonant").nf
        path = str(tmp_path / "nf.json")
        nf.to_json(path, indent=2)
        assert "\n" in open(path).read()
        assert_same_results(nf, NormalForm4D.from_json(path))

    def test_file_content(self, tmp_path):
        c = get_case("quasiresonant_near_tune")
        c.nf.to_json(tmp_path / "nf.json")
        data = json.loads((tmp_path / "nf.json").read_text())
        assert data["format"] == "xnf4d.NormalForm4D"
        assert data["format_version"] == 1
        assert data["settings"] == {
            "max_nf_order": 5,
            "res_case": "quasiresonant",
            "res_eig": [[v.real, v.imag] for v in eigs(1 / 3, QY)],
            "res_basis1": [3, 0],
            "res_basis2": None,
        }
        assert set(data["results"]) == {"Phi", "Psi", "N", "U", "H"}
        # the map is stored truncated at the normal form order
        stored = Map.from_dict(data["complex_map"])
        assert stored.get_max_order() == 5

    def test_transformations_after_loading(self, tmp_path, rng):
        nf = get_case("resonant_exact_tune").nf
        nf.to_json(tmp_path / "nf.json")
        loaded = NormalForm4D.from_json(tmp_path / "nf.json")
        z = real_to_complex(*(rng.normal(size=(4, 5)) * 0.02))
        zeta = loaded.norm_to_nf(*z)
        for a, b in zip(zeta, nf.norm_to_nf(*z)):
            np.testing.assert_array_equal(a, b)
        for a, b in zip(loaded.nf_to_norm(*zeta), nf.nf_to_norm(*zeta)):
            np.testing.assert_array_equal(a, b)

    def test_recompute_after_loading(self, case):
        loaded = NormalForm4D.from_dict(json.loads(json.dumps(case.nf.to_dict())))
        loaded.compute_normal_form()
        for name in ("Phi", "Psi", "N", "U"):
            assert getattr(loaded, name) == getattr(case.nf, name)
        assert loaded.H == case.nf.H

    def test_not_computed(self, tmp_path):
        nf = NormalForm4D(real_map_to_complex(henon_real(QX, QY, 0.7, 0.4), 3), 4)
        nf.to_json(tmp_path / "nf.json")
        loaded = NormalForm4D.from_json(tmp_path / "nf.json")
        assert all(
            getattr(loaded, name) is None for name in ("Phi", "Psi", "N", "U", "H")
        )
        loaded.compute_normal_form()
        nf.compute_normal_form()
        assert loaded.H == nf.H

    def test_unknown_format(self):
        data = get_case("nonresonant").nf.to_dict()
        with pytest.raises(ValueError, match="does not describe"):
            NormalForm4D.from_dict({**data, "format": "something else"})
        with pytest.raises(ValueError, match="does not describe"):
            NormalForm4D.from_dict([1, 2])

    @pytest.mark.parametrize("version", [2, 0, None, "1"])
    def test_unsupported_version(self, version):
        data = get_case("nonresonant").nf.to_dict()
        with pytest.raises(ValueError, match="format version"):
            NormalForm4D.from_dict({**data, "format_version": version})

    @pytest.mark.parametrize("missing", ["settings", "complex_map", "results"])
    def test_missing_keys(self, missing):
        data = get_case("nonresonant").nf.to_dict()
        del data[missing]
        with pytest.raises(ValueError, match="Invalid"):
            NormalForm4D.from_dict(data)

    def test_missing_result(self):
        data = get_case("nonresonant").nf.to_dict()
        del data["results"]["U"]
        with pytest.raises(ValueError, match="Invalid"):
            NormalForm4D.from_dict(data)

    def test_gzip_file_written_by_hand(self, tmp_path):
        nf = get_case("nonresonant").nf
        with gzip.open(tmp_path / "nf.gz", "wt") as f:
            json.dump(nf.to_dict(), f)
        assert_same_results(nf, NormalForm4D.from_json(tmp_path / "nf.gz"))
