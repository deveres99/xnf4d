from __future__ import annotations

import itertools

import numpy as np
import pytest

from xnf4d.polynom import Map, Polynom, Term

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)


def all_exponents(max_order: int, min_order: int = 0):
    """All 4-tuples of non-negative exponents with order in [min, max]."""
    for e in itertools.product(range(max_order + 1), repeat=4):
        if min_order <= sum(e) <= max_order:
            yield e


def random_polynom(rng, max_order, n_terms=None, min_order=0, scale=1.0) -> Polynom:
    """Random collected polynomial with complex coefficients."""
    exps = list(all_exponents(max_order, min_order))
    if n_terms is not None:
        idx = rng.choice(len(exps), size=min(n_terms, len(exps)), replace=False)
        exps = [exps[i] for i in idx]
    return Polynom(
        [
            Term(complex(rng.normal(), rng.normal()) * scale ** sum(e), *e)
            for e in exps
        ]
    )


def random_map(rng, max_order, n_terms=None, min_order=0, scale=1.0) -> Map:
    return Map(
        *(random_polynom(rng, max_order, n_terms, min_order, scale) for _ in range(4))
    )


def identity_map() -> Map:
    return Map(
        Polynom([Term(1.0, 1, 0, 0, 0)]),
        Polynom([Term(1.0, 0, 1, 0, 0)]),
        Polynom([Term(1.0, 0, 0, 1, 0)]),
        Polynom([Term(1.0, 0, 0, 0, 1)]),
    )


def random_points(rng, n=7, scale=0.5):
    return [scale * (rng.normal(size=n) + 1j * rng.normal(size=n)) for _ in range(4)]


def as_dict(poly) -> dict:
    """Polynomial as {exponents: coeff}, summing duplicates."""
    out: dict = {}
    for t in poly.terms:
        out[t.exponents] = out.get(t.exponents, 0) + t.coeff
    return out


def assert_polys_close(p, q, rtol=1e-12, atol=1e-12):
    dp, dq = as_dict(p), as_dict(q)
    for key in set(dp) | set(dq):
        np.testing.assert_allclose(
            dp.get(key, 0), dq.get(key, 0), rtol=rtol, atol=atol, err_msg=str(key)
        )


# --------------------------------------------------------------------------- #
# Term
# --------------------------------------------------------------------------- #


class TestTerm:
    def test_defaults(self):
        t = Term(2.0)
        assert t.exponents == (0, 0, 0, 0)
        assert t.order == 0

    def test_exponents_and_order(self):
        t = Term(1.0, 1, 2, 3, 4)
        assert (t.exp0, t.exp1, t.exp2, t.exp3) == (1, 2, 3, 4)
        assert t.exponents == (1, 2, 3, 4)
        assert t.order == 10

    def test_keyword_construction(self):
        t = Term(coeff=1j, exp0=1, exp3=2)
        assert t.exponents == (1, 0, 0, 2)

    def test_str(self):
        assert str(Term(1.5, 1, 0, 2, 3)) == "1.5\t( 1 0 2 3 )"

    def test_eq_identical(self):
        assert Term(1.0 + 2j, 1, 2, 0, 1) == Term(1.0 + 2j, 1, 2, 0, 1)

    def test_eq_int_float_complex(self):
        assert Term(1, 1) == Term(1.0, 1)
        assert Term(1.0, 1) == Term(1.0 + 0j, 1)

    def test_eq_within_relative_tolerance(self):
        assert Term(1.0, 1) == Term(1.0 + 1e-15, 1)
        assert Term(1e5j, 1) == Term((1e5 + 1e-10) * 1j, 1)

    def test_eq_different_coeff(self):
        assert Term(1.0, 1) != Term(1.0 + 1e-8, 1)
        assert Term(1.0 + 1j, 1) != Term(1.0 + 1.1j, 1)
        assert Term(1.0, 1) != Term(1.0j, 1)

    def test_eq_small_coefficients_scaled_tolerance(self):
        # The absolute tolerance scales with the magnitude of the values.
        assert Term(1e-8, 1) == Term(1e-8 * (1 + 1e-15), 1)
        assert Term(1e-8, 1) != Term(1.1e-8, 1)
        assert Term(1e-20, 1) == Term(0.0, 1)

    @pytest.mark.parametrize("var", range(4))
    def test_eq_different_exponents(self, var):
        e = [0, 0, 0, 0]
        e[var] = 1
        assert Term(1.0) != Term(1.0, *e)

    def test_eq_other_type(self):
        assert (Term(1.0) == 1.0) is False
        assert Term(1.0) != "a"
        assert Term(1.0).__eq__(1.0) is NotImplemented

    def test_hash_consistent_with_eq(self):
        a, b = Term(0.1 + 0.2j, 1, 0, 2, 0), Term(0.1 + 0.2j, 1, 0, 2, 0)
        assert hash(a) == hash(b)
        assert len({a, b}) == 1
        assert hash(Term(1.0, 1)) != hash(Term(1.0, 0, 1))

    def test_product(self):
        p = Term.product_Terms(Term(2.0, 1, 0, 1, 0), Term(3j, 0, 2, 1, 1), 10)
        assert p.coeff == 6j
        assert p.exponents == (1, 2, 2, 1)

    def test_product_at_max_order_kept(self):
        p = Term.product_Terms(Term(2.0, 1), Term(3.0, 0, 1), 2)
        assert p == Term(6.0, 1, 1)

    def test_product_truncated(self):
        p = Term.product_Terms(Term(2.0, 1, 1), Term(3.0, 0, 1), 2)
        assert p.coeff == 0
        assert p.exponents == (0, 0, 0, 0)

    @pytest.mark.parametrize("zero", [0.0, 1e-17, 1e-17j])
    def test_product_zero_coeff(self, zero):
        assert Term.product_Terms(Term(zero, 1), Term(5.0, 1), 5).coeff == 0
        assert Term.product_Terms(Term(5.0, 1), Term(zero, 1), 5).coeff == 0

    def test_power(self):
        p = Term.power_Term(Term(2.0j, 1, 0, 2, 0), 3, 10)
        assert p.coeff == (2.0j) ** 3
        assert p.exponents == (3, 0, 6, 0)

    def test_power_truncated(self):
        assert Term.power_Term(Term(2.0, 1, 1), 3, 5).coeff == 0

    def test_power_zero_coeff(self):
        assert Term.power_Term(Term(0.0, 1), 2, 5).coeff == 0

    def test_power_zero(self):
        p = Term.power_Term(Term(3.0, 1, 2), 0, 5)
        assert p == Term(1.0)


# --------------------------------------------------------------------------- #
# Polynom
# --------------------------------------------------------------------------- #


class TestPolynom:
    def test_str(self):
        p = Polynom([Term(1.0, 1), Term(2.0, 0, 1)])
        assert str(p) == "1.0\t( 1 0 0 0 )\n2.0\t( 0 1 0 0 )"

    def test_eq_independent_of_order(self, rng):
        p = random_polynom(rng, 3)
        q = Polynom(p.terms[::-1])
        assert p == q

    def test_eq_tolerance(self):
        assert Polynom([Term(1.0, 1)]) == Polynom([Term(1.0 + 1e-15, 1)])

    def test_eq_differences(self):
        p = Polynom([Term(1.0, 1), Term(2.0, 0, 1)])
        assert p != Polynom([Term(1.0, 1)])
        assert p != Polynom([Term(1.0, 1), Term(2.5, 0, 1)])
        assert p != Polynom([Term(1.0, 1), Term(2.0, 0, 0, 1)])
        assert p != Polynom([Term(1.0, 1), Term(1.0, 1)])

    def test_eq_duplicates_matched_once(self):
        p = Polynom([Term(1.0, 1), Term(1.0, 1)])
        assert p == Polynom([Term(1.0, 1), Term(1.0, 1)])
        assert p != Polynom([Term(1.0, 1), Term(2.0, 1)])

    def test_eq_empty(self):
        assert Polynom([]) == Polynom([])

    def test_eq_other_type(self):
        assert (Polynom([]) == []) is False
        assert Polynom([]).__eq__(0) is NotImplemented

    def test_truncate_at_order(self, rng):
        p = random_polynom(rng, 5)
        p.truncate_at_order(2)
        assert p.get_max_order() == 2
        assert len(p.terms) == len(list(all_exponents(2)))

    def test_remove_zero_terms(self):
        p = Polynom([Term(0.0, 1), Term(1e-17j, 2), Term(1e-15, 3), Term(1.0)])
        p.remove_zero_terms()
        assert [t.exponents for t in p.terms] == [(3, 0, 0, 0), (0, 0, 0, 0)]

    def test_collect_terms(self):
        a, b, c = Term(1.0, 1), Term(2.0, 0, 1), Term(3.0j, 1)
        p = Polynom([a, b, c])
        p.collect_terms()
        assert p == Polynom([Term(1.0 + 3.0j, 1), Term(2.0, 0, 1)])
        # first-appearance order is kept
        assert [t.exponents for t in p.terms] == [(1, 0, 0, 0), (0, 1, 0, 0)]
        # original terms are left untouched
        assert a.coeff == 1.0 and c.coeff == 3.0j

    def test_substitute_scalar(self):
        p = Polynom([Term(2.0, 1, 0, 0, 0), Term(-1.0j, 0, 2, 1, 0), Term(3.0)])
        v = p.substitute(2.0, 3.0, 0.5, 7.0)
        assert v == pytest.approx(2.0 * 2.0 - 1.0j * 9.0 * 0.5 + 3.0)

    def test_substitute_array(self, rng):
        p = random_polynom(rng, 4)
        pts = random_points(rng)
        expected = sum(
            t.coeff * pts[0] ** t.exp0 * pts[1] ** t.exp1 * pts[2] ** t.exp2 * pts[3] ** t.exp3
            for t in p.terms
        )
        np.testing.assert_allclose(p.substitute(*pts), expected, rtol=1e-13)

    def test_substitute_broadcast(self):
        p = Polynom([Term(1.0, 1, 1)])
        out = p.substitute(np.arange(3.0)[:, None], np.arange(2.0)[None, :], 0.0, 0.0)
        np.testing.assert_allclose(out, np.outer(np.arange(3.0), np.arange(2.0)))

    def test_substitute_empty(self):
        assert Polynom([]).substitute(1.0, 2.0, 3.0, 4.0) == 0

    def test_get_max_order(self):
        assert Polynom([]).get_max_order() == 0
        assert Polynom([Term(1.0, 1), Term(1.0, 1, 2, 0, 1)]).get_max_order() == 4

    def test_product_coeff(self, rng):
        p = random_polynom(rng, 3)
        q = Polynom.product_Coeff_Polynom(2.0 - 1j, p)
        pts = random_points(rng)
        np.testing.assert_allclose(q.substitute(*pts), (2.0 - 1j) * p.substitute(*pts))
        assert q.terms[0] is not p.terms[0]

    def test_sum(self, rng):
        p, q = random_polynom(rng, 3, 10), random_polynom(rng, 3, 10)
        s = Polynom.sum_Polynoms(p, q)
        pts = random_points(rng)
        np.testing.assert_allclose(
            s.substitute(*pts), p.substitute(*pts) + q.substitute(*pts), rtol=1e-13
        )
        assert len({t.exponents for t in s.terms}) == len(s.terms)

    def test_sum_drops_zero_input_terms(self):
        s = Polynom.sum_Polynoms(Polynom([Term(0.0, 1)]), Polynom([Term(1.0, 2)]))
        assert s == Polynom([Term(1.0, 2)])

    def test_sum_does_not_alias_inputs(self):
        p, q = Polynom([Term(1.0, 1)]), Polynom([Term(2.0, 2)])
        s = Polynom.sum_Polynoms(p, q)
        s.terms[0].coeff = 100.0
        assert p.terms[0].coeff == 1.0

    def test_sum_with_empty(self, rng):
        p = random_polynom(rng, 2)
        assert Polynom.sum_Polynoms(p, Polynom([])) == p
        assert Polynom.sum_Polynoms(Polynom([]), p) == p

    def test_product_exact(self, rng):
        p, q = random_polynom(rng, 3), random_polynom(rng, 2)
        r = Polynom.product_Polynoms(p, q, 5)
        pts = random_points(rng)
        np.testing.assert_allclose(
            r.substitute(*pts), p.substitute(*pts) * q.substitute(*pts), rtol=1e-12
        )

    def test_product_simple(self):
        # (1 + v0) * (1 - v0) = 1 - v0**2
        p = Polynom([Term(1.0), Term(1.0, 1)])
        q = Polynom([Term(1.0), Term(-1.0, 1)])
        r = Polynom.product_Polynoms(p, q, 5)
        assert r == Polynom([Term(1.0), Term(0.0, 1), Term(-1.0, 2)])

    @pytest.mark.parametrize("max_order", [0, 1, 2, 3, 4])
    def test_product_truncation(self, rng, max_order):
        p, q = random_polynom(rng, 3), random_polynom(rng, 2)
        full = Polynom.product_Polynoms(p, q, 100)
        full.truncate_at_order(max_order)
        trunc = Polynom.product_Polynoms(p, q, max_order)
        assert trunc.get_max_order() <= max_order
        assert_polys_close(trunc, full)

    def test_product_commutative(self, rng):
        p, q = random_polynom(rng, 3), random_polynom(rng, 3)
        assert_polys_close(
            Polynom.product_Polynoms(p, q, 4), Polynom.product_Polynoms(q, p, 4)
        )

    def test_product_ignores_zero_terms(self):
        p = Polynom([Term(0.0, 1), Term(1.0, 0, 1)])
        r = Polynom.product_Polynoms(p, Polynom([Term(2.0)]), 4)
        assert r == Polynom([Term(2.0, 0, 1)])

    def test_product_with_empty(self, rng):
        assert Polynom.product_Polynoms(random_polynom(rng, 2), Polynom([]), 4) == Polynom([])

    def test_product_handles_uncollected_inputs(self):
        p = Polynom([Term(1.0, 1), Term(2.0, 1)])  # 3 v0, written twice
        r = Polynom.product_Polynoms(p, p, 4)
        assert r == Polynom([Term(9.0, 2)])

    @pytest.mark.parametrize("pow", [2, 3, 4])
    def test_power(self, rng, pow):
        p = random_polynom(rng, 2, 6)
        r = Polynom.power_Polynom(p, pow, 2 * pow)
        pts = random_points(rng)
        np.testing.assert_allclose(r.substitute(*pts), p.substitute(*pts) ** pow, rtol=1e-11)

    def test_power_matches_repeated_product(self, rng):
        p = random_polynom(rng, 2)
        expected = Polynom.product_Polynoms(
            Polynom.product_Polynoms(p, p, 4), p, 4
        )
        assert Polynom.power_Polynom(p, 3, 4) == expected

    def test_power_zero_and_one(self, rng):
        p = random_polynom(rng, 2)
        assert Polynom.power_Polynom(p, 0, 3) == Polynom([Term(1)])
        assert Polynom.power_Polynom(Polynom([]), 0, 3) == Polynom([Term(1)])
        assert Polynom.power_Polynom(p, 1, 3) is p

    def test_power_negative_raises(self):
        with pytest.raises(ValueError):
            Polynom.power_Polynom(Polynom([Term(1.0, 1)]), -1, 3)

    def test_power_binomial(self):
        # (v0 + v1)**3 with binomial coefficients
        p = Polynom([Term(1.0, 1), Term(1.0, 0, 1)])
        r = Polynom.power_Polynom(p, 3, 3)
        assert r == Polynom(
            [Term(1.0, 3), Term(3.0, 2, 1), Term(3.0, 1, 2), Term(1.0, 0, 3)]
        )
        assert Polynom.power_Polynom(p, 3, 2) == Polynom([])

    @pytest.mark.parametrize("var", range(4))
    def test_diff(self, var):
        p = Polynom([Term(2.0, 1, 2, 3, 4), Term(5.0), Term(1j, 0, 1, 0, 0)])
        d = Polynom.diff_Polynom(p, var)
        exps = [1, 2, 3, 4]
        n = exps[var]
        exps[var] -= 1
        expected = [Term(2.0 * n, *exps)]
        if var == 1:
            expected.append(Term(1j))
        assert d == Polynom(expected)

    @pytest.mark.parametrize("var", range(4))
    def test_diff_numerically(self, rng, var):
        p = random_polynom(rng, 4)
        d = Polynom.diff_Polynom(p, var)
        pts = [np.array([0.3 + 0.1j]), np.array([-0.2]), np.array([0.5j]), np.array([0.1])]
        h = 1e-6
        plus = [x.copy() for x in pts]
        minus = [x.copy() for x in pts]
        plus[var] = plus[var] + h
        minus[var] = minus[var] - h
        fd = (p.substitute(*plus) - p.substitute(*minus)) / (2 * h)
        np.testing.assert_allclose(d.substitute(*pts), fd, rtol=1e-7, atol=1e-8)

    @pytest.mark.parametrize("var", ["x", -1, 4, 1.0, True])
    def test_diff_invalid_var(self, var):
        with pytest.raises(ValueError):
            Polynom.diff_Polynom(Polynom([Term(1.0, 1)]), var)

    def test_diff_accepts_numpy_int(self):
        d = Polynom.diff_Polynom(Polynom([Term(1.0, 0, 0, 2)]), np.int64(2))
        assert d == Polynom([Term(2.0, 0, 0, 1)])


# --------------------------------------------------------------------------- #
# Map
# --------------------------------------------------------------------------- #


class TestMap:
    def test_polys(self, rng):
        m = random_map(rng, 2)
        assert m.polys == (m.poly0, m.poly1, m.poly2, m.poly3)

    def test_str(self):
        s = str(identity_map())
        assert s.startswith("poly0:\n1.0\t( 1 0 0 0 )\npoly1:")
        assert "poly3:\n1.0\t( 0 0 0 1 )" in s

    def test_eq(self, rng):
        m = random_map(rng, 2)
        m2 = Map(*(Polynom(p.terms[::-1]) for p in m.polys))
        assert m == m2
        assert m != identity_map()
        assert (m == 1) is False

    def test_substitute(self, rng):
        m = random_map(rng, 3)
        pts = random_points(rng)
        out = m.substitute(*pts)
        assert len(out) == 4
        for o, p in zip(out, m.polys):
            np.testing.assert_allclose(o, p.substitute(*pts))

    def test_substitute_identity(self, rng):
        pts = random_points(rng)
        for o, x in zip(identity_map().substitute(*pts), pts):
            np.testing.assert_allclose(o, x)

    def test_get_max_order(self):
        m = Map(
            Polynom([Term(1.0, 1)]),
            Polynom([Term(1.0, 0, 3)]),
            Polynom([]),
            Polynom([Term(1.0, 1, 1)]),
        )
        assert m.get_max_order() == 3
        assert isinstance(m.get_max_order(), int)

    def test_composition_identity(self, rng):
        m = random_map(rng, 3)
        assert Map.composition_Map(identity_map(), m, 3) == m
        assert Map.composition_Map(m, identity_map(), 3) == m

    def test_composition_exact(self, rng):
        # Without truncation the composition equals successive substitution.
        m1 = random_map(rng, 2, n_terms=6)
        m2 = random_map(rng, 2, n_terms=6)
        comp = Map.composition_Map(m1, m2, 4)
        pts = random_points(rng)
        expected = m2.substitute(*m1.substitute(*pts))
        for o, e in zip(comp.substitute(*pts), expected):
            np.testing.assert_allclose(o, e, rtol=1e-11)

    def test_composition_simple(self):
        # outer: v0 -> v0**2 + v1 ; inner: v0 -> v0 + v1, v1 -> 2 v1
        inner = Map(
            Polynom([Term(1.0, 1), Term(1.0, 0, 1)]),
            Polynom([Term(2.0, 0, 1)]),
            Polynom([Term(1.0, 0, 0, 1)]),
            Polynom([Term(1.0, 0, 0, 0, 1)]),
        )
        outer = Map(
            Polynom([Term(1.0, 2), Term(1.0, 0, 1)]),
            Polynom([Term(1.0, 0, 1)]),
            Polynom([Term(1.0, 0, 0, 1)]),
            Polynom([Term(1.0, 0, 0, 0, 1)]),
        )
        comp = Map.composition_Map(inner, outer, 2)
        # (v0 + v1)**2 + 2 v1
        assert comp.poly0 == Polynom(
            [Term(1.0, 2), Term(2.0, 1, 1), Term(1.0, 0, 2), Term(2.0, 0, 1)]
        )
        assert comp.poly1 == Polynom([Term(2.0, 0, 1)])
        # truncated at first order only the linear part survives
        assert Map.composition_Map(inner, outer, 1).poly0 == Polynom([Term(2.0, 0, 1)])

    @pytest.mark.parametrize("max_order", [1, 2, 3])
    def test_composition_truncation(self, rng, max_order):
        # For maps without constant terms, truncating the composition is the
        # same as truncating the exact composition.
        m1 = random_map(rng, 3, n_terms=8, min_order=1)
        m2 = random_map(rng, 3, n_terms=8, min_order=1)
        full = Map.composition_Map(m1, m2, 9)
        trunc = Map.composition_Map(m1, m2, max_order)
        for f, t in zip(full.polys, trunc.polys):
            f.truncate_at_order(max_order)
            assert_polys_close(t, f)

    def test_composition_associative(self, rng):
        a, b, c = (random_map(rng, 3, n_terms=8, min_order=1) for _ in range(3))
        order = 3
        left = Map.composition_Map(Map.composition_Map(a, b, order), c, order)
        right = Map.composition_Map(a, Map.composition_Map(b, c, order), order)
        for p, q in zip(left.polys, right.polys):
            assert_polys_close(p, q, rtol=1e-10, atol=1e-10)

    def test_composition_cancellation(self):
        # outer v0 - v1 with inner v0 -> v2, v1 -> v2 cancels exactly
        inner = Map(
            Polynom([Term(1.0, 0, 0, 1)]),
            Polynom([Term(1.0, 0, 0, 1)]),
            Polynom([Term(1.0, 0, 0, 1)]),
            Polynom([Term(1.0, 0, 0, 0, 1)]),
        )
        outer = Map(
            Polynom([Term(1.0, 1), Term(-1.0, 0, 1)]),
            Polynom([Term(1.0, 1), Term(-1.0, 0, 1), Term(3.0, 0, 0, 1)]),
            Polynom([]),
            Polynom([Term(1.0, 0, 0, 0, 1)]),
        )
        comp = Map.composition_Map(inner, outer, 3)
        assert comp.poly0 == Polynom([])
        assert comp.poly1 == Polynom([Term(3.0, 0, 0, 1)])
        assert comp.poly2 == Polynom([])

    def test_composition_with_constant_terms(self, rng):
        m1 = random_map(rng, 2, n_terms=5)
        m2 = random_map(rng, 2, n_terms=5)
        comp = Map.composition_Map(m1, m2, 4)
        pts = random_points(rng)
        for o, e in zip(comp.substitute(*pts), m2.substitute(*m1.substitute(*pts))):
            np.testing.assert_allclose(o, e, rtol=1e-11)

    def test_composition_with_empty_component(self, rng):
        m1 = random_map(rng, 2, n_terms=5)
        m1.poly2 = Polynom([])
        m2 = random_map(rng, 2, n_terms=5)
        comp = Map.composition_Map(m1, m2, 4)
        pts = random_points(rng)
        for o, e in zip(comp.substitute(*pts), m2.substitute(*m1.substitute(*pts))):
            np.testing.assert_allclose(o, e, rtol=1e-11)


# --------------------------------------------------------------------------- #
# Serialisation
# --------------------------------------------------------------------------- #


class TestSerialisation:
    def test_polynom_roundtrip_exact(self, rng):
        p = random_polynom(rng, 4)
        p.terms.append(Term(-0.0 + 1e-300j, 0, 0, 0, 7))
        q = Polynom.from_dict(p.to_dict())
        assert [t.exponents for t in q.terms] == [t.exponents for t in p.terms]
        assert [t.coeff for t in q.terms] == [complex(t.coeff) for t in p.terms]

    def test_polynom_json_compatible(self, rng):
        import json

        p = random_polynom(rng, 3)
        q = Polynom.from_dict(json.loads(json.dumps(p.to_dict())))
        assert [t.coeff for t in q.terms] == [t.coeff for t in p.terms]

    def test_real_and_numpy_coefficients(self):
        p = Polynom([Term(2, 1), Term(1.5, 0, 1), Term(np.complex128(1 + 2j), 0, 0, 1)])
        d = p.to_dict()
        assert d == {
            "exponents": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
            "coeff_real": [2.0, 1.5, 1.0],
            "coeff_imag": [0.0, 0.0, 2.0],
        }
        assert all(type(v) is float for v in d["coeff_real"] + d["coeff_imag"])
        assert Polynom.from_dict(d) == p

    def test_empty(self):
        assert Polynom.from_dict(Polynom([]).to_dict()) == Polynom([])

    @pytest.mark.parametrize(
        "data",
        [
            {},
            None,
            {"exponents": [[1, 0, 0, 0]], "coeff_real": [1.0], "coeff_imag": []},
            {"exponents": [[1, 0, 0]], "coeff_real": [1.0], "coeff_imag": [0.0]},
        ],
    )
    def test_invalid_polynom_data(self, data):
        with pytest.raises(ValueError):
            Polynom.from_dict(data)

    def test_map_roundtrip(self, rng):
        m = random_map(rng, 3)
        d = m.to_dict()
        assert list(d) == ["poly0", "poly1", "poly2", "poly3"]
        m2 = Map.from_dict(d)
        for p, q in zip(m.polys, m2.polys):
            assert [t.coeff for t in q.terms] == [t.coeff for t in p.terms]

    def test_invalid_map_data(self, rng):
        d = random_map(rng, 2).to_dict()
        del d["poly3"]
        with pytest.raises(ValueError):
            Map.from_dict(d)
