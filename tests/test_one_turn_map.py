from __future__ import annotations

import os
import warnings
from math import comb, factorial

import numpy as np
import pytest

from xnf4d.one_turn_map import OneTurnMap4D, _LieMapModel, _monomial_values, _monomials
from xnf4d.polynom import Map, Polynom, Term

QX, QY = 0.31, 0.32


def rotation(qx: float = QX, qy: float = QY) -> np.ndarray:
    a, b = 2 * np.pi * qx, 2 * np.pi * qy
    R = np.zeros((4, 4))
    R[:2, :2] = [[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]]
    R[2:, 2:] = [[np.cos(b), np.sin(b)], [-np.sin(b), np.cos(b)]]
    return R


# --------------------------------------------------------------------------- #
# Reference implementation with the polynomial algebra
# --------------------------------------------------------------------------- #


def poisson(f: Polynom, g: Polynom, order: int) -> Polynom:
    out = Polynom([])
    for q, p in ((0, 1), (2, 3)):
        out = Polynom.sum_Polynoms(
            out,
            Polynom.product_Polynoms(
                Polynom.diff_Polynom(f, q), Polynom.diff_Polynom(g, p), order
            ),
        )
        out = Polynom.sum_Polynoms(
            out,
            Polynom.product_Coeff_Polynom(
                -1.0,
                Polynom.product_Polynoms(
                    Polynom.diff_Polynom(f, p), Polynom.diff_Polynom(g, q), order
                ),
            ),
        )
    return out


def lie_exp(f: Polynom, g: Polynom, order: int) -> Polynom:
    res, cur = g, g
    for n in range(1, order + 1):
        cur = poisson(f, cur, order)
        res = Polynom.sum_Polynoms(res, Polynom.product_Coeff_Polynom(1 / factorial(n), cur))
    return res


def reference_map(R: np.ndarray, generators: list[Polynom], order: int) -> Map:
    polys = []
    for i in range(4):
        g = Polynom([Term(R[i, j], *[int(j == k) for k in range(4)]) for j in range(4)])
        for f in generators:
            g = lie_exp(f, g, order)
        polys.append(g)
    return Map(*polys)


def as_dict(poly: Polynom) -> dict:
    out: dict = {}
    for t in poly.terms:
        out[t.exponents] = out.get(t.exponents, 0) + t.coeff
    return out


def max_diff(p: Polynom, q: Polynom) -> float:
    a, b = as_dict(p), as_dict(q)
    return max([abs(a.get(k, 0) - b.get(k, 0)) for k in set(a) | set(b)] + [0.0])


def assert_symplectic(m: Map, order: int, atol: float) -> None:
    J = {(0, 1): 1.0, (2, 3): 1.0}
    for a in range(4):
        for b in range(a + 1, 4):
            br = poisson(m.polys[a], m.polys[b], order)
            br.truncate_at_order(order)
            expected = Polynom([Term(J.get((a, b), 0.0))])
            assert max_diff(br, expected) < atol, (a, b)


# --------------------------------------------------------------------------- #
# Monomials and the Lie map model
# --------------------------------------------------------------------------- #


class TestMonomials:
    @pytest.mark.parametrize("order", [1, 3, 6])
    def test_counts(self, order):
        exps = _monomials(1, order)
        assert len(exps) == comb(order + 4, 4) - 1
        assert len({tuple(e) for e in exps.tolist()}) == len(exps)
        assert np.all(np.diff(exps.sum(axis=1)) >= 0)
        assert len(_monomials(order, order)) == comb(order + 3, 3)

    def test_values(self, rng):
        z = rng.normal(size=(4, 7))
        exps = _monomials(1, 3)
        V = _monomial_values(z, exps)
        expected = np.prod(z.T[:, None, :] ** exps[None, :, :], axis=2)
        np.testing.assert_allclose(V, expected, rtol=1e-13)


@pytest.fixture
def rng():
    return np.random.default_rng(7)


class TestLieMapModel:
    def test_sizes(self):
        m = _LieMapModel(rotation(), 5)
        assert m.degrees == [3, 4, 5, 6]
        assert m.n_mono == comb(9, 4) - 1
        assert m.n_params == sum(comb(k + 3, 3) for k in range(3, 7))

    def test_invalid_order(self):
        with pytest.raises(ValueError):
            _LieMapModel(rotation(), 1)

    def test_zero_generators_give_linear_map(self):
        m = _LieMapModel(rotation(), 4)
        C = m.coefficients(np.zeros(m.n_params))
        np.testing.assert_array_equal(C, m.C0)
        lin = m.to_map(C)
        for i in range(4):
            assert as_dict(lin.polys[i]) == pytest.approx(
                {tuple(int(j == k) for k in range(4)): rotation()[i, j]
                 for j in range(4) if rotation()[i, j] != 0}
            )

    @pytest.mark.parametrize("order", [2, 3, 5])
    def test_matches_lie_series(self, rng, order):
        m = _LieMapModel(rotation(), order)
        theta = rng.normal(size=m.n_params) * 0.3
        got = m.to_map(m.coefficients(theta))
        ref = reference_map(rotation(), m.generators(theta), order)
        for p, q in zip(got.polys, ref.polys):
            assert max_diff(p, q) < 1e-13

    @pytest.mark.parametrize("order", [3, 5])
    def test_jacobian_matches_finite_differences(self, rng, order):
        m = _LieMapModel(rotation(), order)
        theta = rng.normal(size=m.n_params) * 0.3
        left = rng.normal(size=(m.n_mono + 3, m.n_mono))
        C, dC = m.coefficients_and_jacobian(theta, left=left)
        np.testing.assert_allclose(C, m.coefficients(theta), atol=1e-14)
        assert dC.shape == (m.n_mono + 3, 4, m.n_params)
        h = 1e-6
        for a in rng.choice(m.n_params, 20, replace=False):
            tp, tm = theta.copy(), theta.copy()
            tp[a] += h
            tm[a] -= h
            fd = left @ (m.coefficients(tp) - m.coefficients(tm)) / (2 * h)
            np.testing.assert_allclose(dC[:, :, a], fd, rtol=1e-7, atol=1e-8)

    def test_jacobian_without_left_matrix(self, rng):
        m = _LieMapModel(rotation(), 4)
        theta = rng.normal(size=m.n_params) * 0.3
        _, dC = m.coefficients_and_jacobian(theta)
        _, dC2 = m.coefficients_and_jacobian(theta, left=np.eye(m.n_mono))
        np.testing.assert_allclose(dC, dC2, atol=1e-14)

    @pytest.mark.parametrize("order", [3, 5])
    def test_map_is_symplectic(self, rng, order):
        m = _LieMapModel(rotation(), order)
        theta = rng.normal(size=m.n_params) * 0.3
        assert_symplectic(m.to_map(m.coefficients(theta)), order - 1, atol=1e-12)

    def test_generators(self, rng):
        m = _LieMapModel(rotation(), 4)
        theta = rng.normal(size=m.n_params)
        gens = m.generators(theta)
        assert [g.get_max_order() for g in gens] == [3, 4, 5]
        assert all(all(t.order == k for t in g.terms) for g, k in zip(gens, (3, 4, 5)))
        assert sum(len(g.terms) for g in gens) == m.n_params


# --------------------------------------------------------------------------- #
# Fit on exact data
# --------------------------------------------------------------------------- #

K2, K3 = 0.3, 0.2


def kick_potential() -> tuple[Polynom, Polynom]:
    """V = K2/3 (x^3 - 3 x y^2) + K3/4 (x^4 - 6 x^2 y^2 + y^4)."""
    v3 = Polynom([Term(K2 / 3, 3, 0, 0, 0), Term(-K2, 1, 0, 2, 0)])
    v4 = Polynom(
        [Term(K3 / 4, 4, 0, 0, 0), Term(-1.5 * K3, 2, 0, 2, 0), Term(K3 / 4, 0, 0, 4, 0)]
    )
    return v3, v4


def kick_then_rotate(z: np.ndarray) -> np.ndarray:
    x, px, y, py = z
    px = px - (K2 * (x**2 - y**2) + K3 * (x**3 - 3 * x * y**2))
    py = py - (-2 * K2 * x * y + K3 * (-3 * x**2 * y + y**3))
    return rotation() @ np.array([x, px, y, py])


def sample(rng, n: int, amp: float = 1.0) -> np.ndarray:
    J = rng.uniform(0, amp**2 / 2, (2, n))
    phi = rng.uniform(0, 2 * np.pi, (2, n))
    r = np.sqrt(2 * J)
    return np.array(
        [r[0] * np.cos(phi[0]), r[0] * np.sin(phi[0]), r[1] * np.cos(phi[1]), r[1] * np.sin(phi[1])]
    )


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(3)
    z0 = sample(rng, 800)
    otm = OneTurnMap4D.from_data(z0, kick_then_rotate(z0), rotation(), 3)
    otm.compute()
    return otm


class TestFitExactMap:
    """
    kick_then_rotate = R o exp(:f3:) o exp(:f4:) with f = -V exactly (the
    kicks commute), so the fit must recover it to rounding.
    """

    def test_reproduces_data(self, fitted):
        assert fitted.fit_result.success
        assert fitted.fit_rms.max() < 1e-13

    def test_recovers_generators(self, fitted):
        v3, v4 = kick_potential()
        f3, f4 = fitted.generators
        assert max_diff(f3, Polynom.product_Coeff_Polynom(-1.0, v3)) < 1e-12
        assert max_diff(f4, Polynom.product_Coeff_Polynom(-1.0, v4)) < 1e-12

    def test_higher_order_generators_vanish(self):
        rng = np.random.default_rng(4)
        z0 = sample(rng, 2000)
        otm = OneTurnMap4D.from_data(z0, kick_then_rotate(z0), rotation(), 5)
        otm.compute()
        assert otm.fit_rms.max() < 1e-12
        for g in otm.generators[2:]:
            assert max((abs(t.coeff) for t in g.terms), default=0) < 1e-9

    def test_map_on_new_points(self, fitted, rng):
        z = sample(rng, 50)
        np.testing.assert_allclose(fitted.track_map(z), kick_then_rotate(z), atol=1e-13)

    def test_track_map_several_turns(self, fitted, rng):
        z = sample(rng, 5, amp=0.3)
        expected = z
        for _ in range(4):
            expected = kick_then_rotate(expected)
        np.testing.assert_allclose(fitted.track_map(z, num_turns=4), expected, atol=1e-12)

    def test_symplectic(self, fitted):
        assert_symplectic(fitted.one_turn_map, 2, atol=1e-12)

    def test_complex_map(self, fitted):
        cm = fitted.complex_one_turn_map
        lam = np.exp(2j * np.pi * QX)
        z1 = {t.exponents: t.coeff for t in cm.poly0.terms}
        assert z1[(1, 0, 0, 0)] == pytest.approx(lam, abs=1e-13)
        # z1* component is the conjugate of the z1 component
        z1s = {t.exponents: t.coeff for t in cm.poly1.terms}
        for (a, b, c, d), v in z1.items():
            assert z1s.get((b, a, d, c), 0) == pytest.approx(np.conj(v), abs=1e-13)

    def test_normal_form(self, fitted):
        nf = fitted.calculate_normal_form(2)
        assert fitted.normal_form is nf
        h = {t.exponents: t.coeff for t in nf.H.terms}
        assert h[(1, 1, 0, 0)] == pytest.approx(2j * np.pi * QX, abs=1e-12)
        assert h[(0, 0, 1, 1)] == pytest.approx(2j * np.pi * QY, abs=1e-12)

    def test_normal_form_order_warning(self, fitted):
        # the map is of order 3: a normal form of order 3 would need order 4
        with pytest.warns(UserWarning, match="map is of order 3"):
            fitted.calculate_normal_form(3)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fitted.calculate_normal_form(2)

    def test_noisy_data(self):
        # With noise the fit is the least-squares optimum, still symplectic
        rng = np.random.default_rng(5)
        z0 = sample(rng, 3000)
        z1 = kick_then_rotate(z0) + 1e-6 * rng.normal(size=z0.shape)
        otm = OneTurnMap4D.from_data(z0, z1, rotation(), 4)
        otm.compute()
        assert 0.5e-6 < otm.fit_rms.max() < 2e-6
        assert_symplectic(otm.one_turn_map, 3, atol=1e-12)


class TestFromDataValidation:
    def test_shapes(self):
        with pytest.raises(ValueError, match="shape"):
            OneTurnMap4D.from_data(np.zeros((3, 100)), np.zeros((3, 100)), rotation(), 3)
        with pytest.raises(ValueError, match="shape"):
            OneTurnMap4D.from_data(np.zeros((4, 100)), np.zeros((4, 90)), rotation(), 3)

    def test_too_few_samples(self, rng):
        z = sample(rng, 20)
        with pytest.raises(ValueError, match="too few"):
            OneTurnMap4D.from_data(z, z, rotation(), 3)

    def test_order(self, rng):
        z = sample(rng, 200)
        with pytest.raises(ValueError, match="max_map_order"):
            OneTurnMap4D.from_data(z, z, rotation(), 1)

    def test_no_line(self, rng):
        z = sample(rng, 200)
        otm = OneTurnMap4D.from_data(z, kick_then_rotate(z), rotation(), 2)
        assert otm.sigma_limit == pytest.approx(np.max(np.abs(z)))
        with pytest.raises(RuntimeError, match="No line"):
            otm.build_particles(z)

    def test_compute_on_demand(self, rng):
        z = sample(rng, 300)
        otm = OneTurnMap4D.from_data(z, kick_then_rotate(z), rotation(), 3)
        assert otm.one_turn_map is None
        otm.complex_one_turn_map
        assert otm.one_turn_map is not None


# --------------------------------------------------------------------------- #
# LHC test line
# --------------------------------------------------------------------------- #

LHC_FILE = os.path.join(
    os.path.dirname(__file__), "..", "test_data", "ramp.0_lhc_b1.json.gz"
)
NEMITT = 2.5e-6


@pytest.fixture(scope="module")
def lhc_line():
    xt = pytest.importorskip("xtrack")
    if not os.path.exists(LHC_FILE):
        pytest.skip("LHC test line not available")
    line = xt.Line.from_json(LHC_FILE)
    line.build_tracker()
    return line


@pytest.fixture(scope="module")
def lhc_map(lhc_line):
    otm = OneTurnMap4D(lhc_line, 4, nemitt_x=NEMITT, nemitt_y=NEMITT, seed=11)
    otm.compute()
    return otm


class TestLHC:
    def test_linear_map_is_rotation(self, lhc_map):
        tw = lhc_map.twiss
        np.testing.assert_allclose(lhc_map.R, rotation(tw.qx % 1, tw.qy % 1), atol=1e-8)
        for i in (0, 2):
            assert np.linalg.det(lhc_map.R[i : i + 2, i : i + 2]) == pytest.approx(1, abs=1e-15)

    def test_normalised_roundtrip(self, lhc_map, rng):
        z = rng.normal(size=(4, 20))
        back = lhc_map.normalised_coordinates(lhc_map.build_particles(z))
        np.testing.assert_allclose(back, z, atol=1e-10)

    def test_particles_on_longitudinal_closed_orbit(self, lhc_map):
        p = lhc_map.build_particles(np.zeros((4, 3)))
        tw = lhc_map.twiss
        np.testing.assert_allclose(p.zeta, tw.zeta[0], atol=1e-12)
        np.testing.assert_allclose(p.delta, tw.delta[0], atol=1e-12)
        np.testing.assert_allclose(p.x, tw.x[0], atol=1e-12)

    def test_sample(self, lhc_map):
        n_mono = comb(8, 4) - 1
        n = max(1000, 10 * n_mono)
        # midplane_symmetry="auto": the mirrored particles follow the sample
        assert lhc_map.z0.shape == lhc_map.z1.shape == (4, 2 * n)
        np.testing.assert_allclose(lhc_map.z0[:, n:], MIRROR * lhc_map.z0[:, :n], atol=1e-9)
        Jx = (lhc_map.z0[0] ** 2 + lhc_map.z0[1] ** 2) / 2
        assert Jx.max() <= lhc_map.sigma_limit**2 / 2 + 1e-9

    def test_fit_accuracy(self, lhc_map):
        assert lhc_map.fit_result.success
        assert lhc_map.fit_rms.max() < 1e-4

    def test_validation_on_new_particles(self, lhc_map):
        z = lhc_map._sample(np.random.default_rng(123), 300)
        z1, alive = lhc_map.track(z)
        assert alive.all()
        err = np.sqrt(np.mean((lhc_map.track_map(z) - z1) ** 2, axis=1))
        assert np.all(err < 3 * lhc_map.fit_rms + 1e-12)

    def test_tracking_one_turn_matches_sample(self, lhc_map):
        z1, _ = lhc_map.track(lhc_map.z0[:, :10])
        np.testing.assert_allclose(z1, lhc_map.z1[:, :10], atol=1e-12)

    def test_fitted_map_symplectic(self, lhc_map):
        assert_symplectic(lhc_map.one_turn_map, 3, atol=1e-12)

    def test_normal_form_tunes(self, lhc_map):
        nf = lhc_map.calculate_normal_form(3)
        h = {t.exponents: t.coeff for t in nf.H.terms}
        tw = lhc_map.twiss
        assert h[(1, 1, 0, 0)].imag == pytest.approx(2 * np.pi * (tw.qx % 1), abs=1e-6)
        assert h[(0, 0, 1, 1)].imag == pytest.approx(2 * np.pi * (tw.qy % 1), abs=1e-6)

    def test_lost_particles(self, lhc_line):
        # far outside the aperture all particles are lost
        with pytest.warns(UserWarning, match="lost"):
            with pytest.raises(ValueError, match="survived"):
                OneTurnMap4D(
                    lhc_line, 2, nemitt_x=NEMITT, nemitt_y=NEMITT, sigma_limit=1e4,
                    num_particles=200, seed=1,
                )

    def test_track_reports_lost(self, lhc_map):
        z = np.array([[0.5, 1e4], [0, 0], [0.5, 0], [0, 0]])
        z1, alive = lhc_map.track(z)
        assert alive.tolist() == [True, False]
        assert np.all(np.isnan(z1[:, 1])) and np.all(np.isfinite(z1[:, 0]))

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            (dict(nemitt_x=0), "emittances"),
            (dict(sigma_limit=-1), "sigma_limit"),
            (dict(method="5d"), "method"),
        ],
    )
    def test_invalid_arguments(self, lhc_line, kwargs, match):
        args = dict(nemitt_x=NEMITT, nemitt_y=NEMITT)
        args.update(kwargs)
        with pytest.raises(ValueError, match=match):
            OneTurnMap4D(lhc_line, 3, **args)


class TestSymplecticRotation:
    def test_exact_rotation_unchanged(self):
        from xnf4d.one_turn_map import _symplectic_rotation

        np.testing.assert_allclose(_symplectic_rotation(rotation()), rotation(), atol=1e-15)

    def test_projection_of_perturbed_matrix(self):
        from xnf4d.one_turn_map import _symplectic_rotation

        R = rotation() * (1 + 3e-10)
        out = _symplectic_rotation(R)
        np.testing.assert_allclose(out, rotation(), atol=1e-14)

    def test_warns_if_not_rotation(self):
        from xnf4d.one_turn_map import _symplectic_rotation

        R = rotation()
        R[0, 2] = 1e-3
        with pytest.warns(UserWarning, match="rotation"):
            _symplectic_rotation(R)


# --------------------------------------------------------------------------- #
# Resonance islands (LHC test line with octupoles)
# --------------------------------------------------------------------------- #

#: Island configurations: tunes (None keeps the nominal ones), octupole
#: current, resonant tune, and horizontal amplitudes [sigma] inside and
#: outside the islands (vertical amplitude 0.3 sigma).
ISLAND_CASES = {
    "third_order": dict(
        tunes=None, i_mo=40, resonance=1 / 3, inside=[5.2, 5.5], outside=[2.0, 4.0, 7.0]
    ),
    "fourth_order": dict(
        tunes=(62.28, 60.31), i_mo=-26, resonance=1 / 4,
        inside=[6.8, 7.5, 8.2], outside=[3.0, 5.0, 5.8],
    ),
}
ISLAND_MAP_ORDER = 8
ISLAND_TURNS = 512


def fractional_tune(x: np.ndarray, px: np.ndarray) -> float:
    """Horizontal tune from turn-by-turn data (Hann window, interpolated FFT peak)."""
    sig = x - 1j * px
    sig = (sig - sig.mean()) * np.hanning(len(sig))
    n_fft = 16 * len(sig)
    spec = np.abs(np.fft.fft(sig, n_fft))
    k = int(np.argmax(spec))
    a, b, c = spec[k - 1], spec[k], spec[(k + 1) % n_fft]
    return float(((k + 0.5 * (a - c) / (a - 2 * b + c)) / n_fft) % 1)


def island_data(line, name: str) -> dict:
    """
    Fitted map and turn-by-turn data (tracking and map) for an island case.
    The knobs of the line are restored before returning.
    """
    case = ISLAND_CASES[name]
    knobs = ("dqx.b1", "dqy.b1", "i_mo_b1")
    saved = {k: line.vars[k]._value for k in knobs}
    try:
        line.vars["i_mo_b1"] = case["i_mo"]
        if case["tunes"] is not None:
            # the tune knobs are linear: two corrections reach the target
            for _ in range(2):
                tw = line.twiss()
                line.vars["dqx.b1"] = line.vars["dqx.b1"]._value + case["tunes"][0] - tw.qx
                line.vars["dqy.b1"] = line.vars["dqy.b1"]._value + case["tunes"][1] - tw.qy
        otm = OneTurnMap4D(
            line, ISLAND_MAP_ORDER, nemitt_x=NEMITT, nemitt_y=NEMITT, seed=2
        )
        otm.compute()
        amps = np.array(case["inside"] + case["outside"])
        z = np.array([amps, 0 * amps, 0.3 + 0 * amps, 0 * amps])
        tracked, mapped = [z], [z]
        for _ in range(ISLAND_TURNS - 1):
            z_next, alive = otm.track(tracked[-1])
            assert alive.all()
            tracked.append(z_next)
            mapped.append(otm.track_map(mapped[-1]))
    finally:
        for k, v in saved.items():
            line.vars[k] = v
    return dict(
        otm=otm,
        case=case,
        tracked=np.array(tracked),
        mapped=np.array(mapped),
        n_inside=len(case["inside"]),
        tunes=(otm.twiss.qx, otm.twiss.qy),
    )


@pytest.fixture(scope="module", params=list(ISLAND_CASES))
def island(request, lhc_line):
    return island_data(lhc_line, request.param)


def tunes_of(traj: np.ndarray) -> np.ndarray:
    return np.array(
        [fractional_tune(traj[:, 0, i], traj[:, 1, i]) for i in range(traj.shape[2])]
    )


class TestIslands:
    """
    The fitted map must reproduce resonance islands, whatever their order:
    particles locked on the resonance in tracking are locked with the map,
    and the others are not.
    """

    def test_configuration(self, island):
        if island["case"]["tunes"] is not None:
            np.testing.assert_allclose(island["tunes"], island["case"]["tunes"], atol=1e-6)

    def test_tracking_shows_islands(self, island):
        q = tunes_of(island["tracked"])
        res, k = island["case"]["resonance"], island["n_inside"]
        assert np.all(np.abs(q[:k] - res) < 1e-5), q
        assert np.all(np.abs(q[k:] - res) > 2e-3), q

    def test_map_reproduces_islands(self, island):
        q = tunes_of(island["mapped"])
        res, k = island["case"]["resonance"], island["n_inside"]
        assert np.all(np.abs(q[:k] - res) < 1e-5), q
        assert np.all(np.abs(q[k:] - res) > 2e-3), q

    def test_map_tunes_outside_islands(self, island):
        k = island["n_inside"]
        q_track = tunes_of(island["tracked"])[k:]
        q_map = tunes_of(island["mapped"])[k:]
        np.testing.assert_allclose(q_map, q_track, atol=1e-3)

    def test_one_turn_accuracy_along_orbits(self, island):
        otm, tracked = island["otm"], island["tracked"]
        err = max(
            np.max(np.abs(otm.track_map(tracked[t]) - tracked[t + 1]))
            for t in range(0, ISLAND_TURNS - 1, 16)
        )
        assert err < 2.5e-2

    def test_map_orbits_stay_bounded(self, island):
        assert np.all(np.isfinite(island["mapped"]))
        amp = np.hypot(island["mapped"][:, 0], island["mapped"][:, 1])
        assert amp.max() < island["otm"].sigma_limit


# --------------------------------------------------------------------------- #
# Midplane symmetry
# --------------------------------------------------------------------------- #

MIRROR = np.array([1.0, 1.0, -1.0, -1.0])[:, None]


def skew_kick_then_rotate(z: np.ndarray) -> np.ndarray:
    """Normal plus skew sextupole kick, from V + a (x^2 y - y^3 / 3): not midplane symmetric."""
    x, px, y, py = z
    a = 0.3 * K2
    px = px - (K2 * (x**2 - y**2) + 2 * a * x * y)
    py = py - (-2 * K2 * x * y + a * (x**2 - y**2))
    return rotation() @ np.array([x, px, y, py])


class TestMidplaneSymmetry:
    def test_model_keeps_even_generators(self):
        full = _LieMapModel(rotation(), 5)
        sym = _LieMapModel(rotation(), 5, midplane_symmetric=True)
        assert sym.n_params < full.n_params
        for k in sym.degrees:
            e = sym.gen_exps[k]
            assert np.all((e[:, 2] + e[:, 3]) % 2 == 0)
            # all even monomials are kept
            ef = full.gen_exps[k]
            assert len(e) == np.sum((ef[:, 2] + ef[:, 3]) % 2 == 0)
        assert sym.n_mono == full.n_mono

    def test_symmetric_map_commutes_with_mirror(self, rng):
        model = _LieMapModel(rotation(), 5, midplane_symmetric=True)
        C = model.coefficients(rng.normal(size=model.n_params) * 0.1)
        z = rng.normal(size=(4, 50)) * 0.5
        V, Vm = _monomial_values(z, model.exps), _monomial_values(MIRROR * z, model.exps)
        np.testing.assert_allclose((Vm @ C).T, MIRROR * (V @ C).T, atol=1e-14)

    def test_from_data_symmetric_fit_of_symmetric_map(self, rng):
        z0 = sample(rng, 800)
        maps = {}
        for opt in (False, True):
            otm = OneTurnMap4D.from_data(z0, kick_then_rotate(z0), rotation(), 4,
                                         midplane_symmetry=opt)
            otm.compute()
            assert otm.midplane_symmetric is opt
            maps[opt] = otm
        z = sample(rng, 50, 0.8)
        np.testing.assert_allclose(maps[True].track_map(z), maps[False].track_map(z), atol=1e-10)
        # a pure horizontal orbit stays in the horizontal plane exactly
        z[2:] = 0
        assert np.all(maps[True].track_map(z, 5)[2:] == 0)

    def test_symmetric_fit_cannot_follow_asymmetric_map(self, rng):
        z0 = sample(rng, 800)
        fits = {}
        for opt in (False, True):
            otm = OneTurnMap4D.from_data(z0, skew_kick_then_rotate(z0), rotation(), 3,
                                         midplane_symmetry=opt)
            otm.compute()
            fits[opt] = otm.fit_rms.max()
        assert fits[False] < 1e-10 < 1e-3 < fits[True]

    @pytest.mark.parametrize("value", ["auto", "yes", 1, None])
    def test_from_data_invalid_option(self, value):
        z = sample(np.random.default_rng(0), 200)
        with pytest.raises(ValueError, match="midplane_symmetry"):
            OneTurnMap4D.from_data(z, z, rotation(), 3, midplane_symmetry=value)

    def test_coupled_linear_map_rejected(self):
        z = sample(np.random.default_rng(0), 200)
        R = rotation()
        R[0, 2] = 1e-3
        with pytest.raises(ValueError, match="couple"):
            OneTurnMap4D.from_data(z, z, R, 3, midplane_symmetry=True)
        OneTurnMap4D.from_data(z, z, R, 3)  # fine without the symmetry


def small_line(ksl2: float = 0.0):
    """Linear map and an octupole, plus an optional skew sextupole."""
    xt = pytest.importorskip("xtrack")
    line = xt.Line(
        elements=[
            xt.LineSegmentMap(length=100.0, betx=100.0, bety=80.0, qx=QX - 0.03, qy=QY,
                              longitudinal_mode="linear_fixed_qs", qs=0.004, bets=500.0),
            xt.Multipole(knl=[0, 0, 0, 300.0], ksl=[0, 0, ksl2, 0]),
        ],
        element_names=["start", "mult"],
    )
    line.set_particle_ref("proton", p0c=450e9)
    line.build_tracker()
    return line


class TestMidplaneSymmetryLine:
    def test_auto_imposes_on_symmetric_line(self):
        otm = OneTurnMap4D(small_line(), 5, 2.5e-6, 2.5e-6, 6, seed=1)
        assert otm.midplane_symmetry == "auto"
        assert otm.midplane_asymmetry < 1e-12
        # the mirrored particles double the data
        assert otm.z0.shape[1] == 2 * max(1000, 10 * otm._model.n_mono)
        otm.compute()
        assert otm.midplane_symmetric is True
        z = np.array([[3.0, 1.0], [0.5, -2.0], [0.0, 0.0], [0.0, 0.0]])
        assert np.all(otm.track_map(z)[2:] == 0)

    def test_auto_rejects_skew_line(self):
        otm = OneTurnMap4D(small_line(ksl2=20.0), 5, 2.5e-6, 2.5e-6, 6, seed=1)
        assert otm.midplane_asymmetry > 1e-6
        otm.compute()
        assert otm.midplane_symmetric is False
        z, _ = otm.track(sample(np.random.default_rng(1), 50, 4.0))
        assert otm.fit_rms.max() < 1e-6

    @pytest.mark.parametrize("value", [True, False])
    def test_explicit_option_does_not_track_mirror(self, value):
        otm = OneTurnMap4D(small_line(), 3, 2.5e-6, 2.5e-6, 6, seed=1, num_particles=1500,
                           midplane_symmetry=value)
        assert otm.midplane_asymmetry is None and otm.z0.shape[1] == 1500
        otm.compute()
        assert otm.midplane_symmetric is value

    def test_invalid_option(self):
        with pytest.raises(ValueError, match="midplane_symmetry"):
            OneTurnMap4D(small_line(), 3, 2.5e-6, 2.5e-6, 6, midplane_symmetry="maybe")
