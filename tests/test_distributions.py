from __future__ import annotations

import os
import warnings

import numpy as np
import pytest

xt = pytest.importorskip("xtrack")

from xnf4d.distributions import (  # noqa: E402
    _cuts,
    _f_2D,
    _f_4D,
    _generate_factorizable,
    _generate_nonfactorizable,
    _normalised_cut,
    _PolyEvaluator,
    _to_complex,
    _to_real,
    _Transforms,
    generate_2D_uniform_circular_sector,
    generate_matched_gaussian_bunch,
    generate_matched_qgaussian_bunch,
    generate_pencil_on_collimator,
    seam_mismatch,
)
from xnf4d.numerical_normal_form import NormalForm4D  # noqa: E402
from xnf4d.one_turn_map import OneTurnMap4D  # noqa: E402
from xnf4d.polynom import Map, Polynom, Term  # noqa: E402

NEMITT = 2.5e-6
QX, QY = 0.28, 0.31
BETX, BETY = 100.0, 80.0
JAW_SIGMAS = 5.0
SPREAD_SIGMAS = 0.1
OCTUPOLE = 300.0
LHC_FILE = os.path.join(
    os.path.dirname(__file__), "..", "test_data", "ramp.0_lhc_b1.json.gz"
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def toy_line(k3: float):
    """Linear one-turn map followed by an octupole kick."""
    line = xt.Line(
        elements=[
            xt.LineSegmentMap(
                length=100.0, betx=BETX, bety=BETY, qx=QX, qy=QY,
                longitudinal_mode="linear_fixed_qs", qs=0.004, bets=500.0,
            ),
            xt.Multipole(knl=[0, 0, 0, k3], ksl=[0, 0, 0, 0]),
        ],
        element_names=["coll", "oct"],
    )
    line.set_particle_ref("proton", p0c=450e9)
    line.build_tracker()
    return line


def linear_normal_form(qx: float = QX, qy: float = QY) -> NormalForm4D:
    """Normal form of a rotation: Psi and Phi are the identity."""
    lam = [np.exp(2j * np.pi * qx), np.exp(-2j * np.pi * qx),
           np.exp(2j * np.pi * qy), np.exp(-2j * np.pi * qy)]
    rot = Map(*(Polynom([Term(lam[j], *(int(i == j) for i in range(4)))]) for j in range(4)))
    nf = NormalForm4D(rot, 2)
    nf.compute_normal_form()
    return nf


def beam_size(twiss, beta: float = BETX) -> float:
    beta0 = float(np.asarray(twiss.particle_on_co.beta0).ravel()[0])
    gamma0 = float(np.asarray(twiss.particle_on_co.gamma0).ravel()[0])
    return float(np.sqrt(beta * NEMITT / (beta0 * gamma0)))


def orbit_depth(line, particles, jaw: float, plane: str = "x", num_turns: int = 2000):
    """Largest distance beyond the jaw of each orbit, tracking from the line start."""
    sign = np.sign(getattr(particles, plane))
    p = particles.copy()
    line.track(p, num_turns=num_turns, turn_by_turn_monitor=True)
    u = getattr(line.record_last_track, plane)
    order = np.argsort(np.asarray(particles.particle_id))
    return np.max(sign[order, None] * u, axis=1) - jaw, p


@pytest.fixture(scope="module")
def linear_setup():
    line = toy_line(0.0)
    tw = line.twiss()
    return line, tw, linear_normal_form(), beam_size(tw)


@pytest.fixture(scope="module")
def octupole_setup():
    line = toy_line(OCTUPOLE)
    otm = OneTurnMap4D(line, 8, NEMITT, NEMITT, sigma_limit=8, seed=1)
    otm.compute()
    tw = otm.twiss
    return line, tw, otm.calculate_normal_form(7), beam_size(tw)


def pencil(setup, num_particles: int = 500, **kwargs):
    line, tw, nf, sig = setup
    args = dict(normal_form=nf, plane="x", jaw=JAW_SIGMAS * sig, nemitt_x=NEMITT,
                nemitt_y=NEMITT, pencil_spread=SPREAD_SIGMAS * sig, twiss=tw, seed=2)
    args.update(kwargs)
    return generate_pencil_on_collimator(line, "coll", num_particles, **args)


# --------------------------------------------------------------------------- #
# Fast evaluation and inversion of the normal form transformations
# --------------------------------------------------------------------------- #


class TestTransforms:
    def test_evaluator_matches_substitute(self, octupole_setup):
        nf = octupole_setup[2]
        z = np.random.default_rng(0).normal(size=(4, 500)) * 3 + 0j
        expected = np.array(nf.Psi.substitute(*z))
        np.testing.assert_allclose(_PolyEvaluator(nf.Psi.polys)(z), expected, rtol=1e-13, atol=1e-13)

    def test_evaluator_shapes_and_blocks(self):
        polys = [Polynom([Term(2.0, 1, 0, 0, 0), Term(1.0, 0, 0, 0, 3)]), Polynom([Term(1.0, 0, 0, 0, 0)])]
        ev = _PolyEvaluator(polys)
        z = np.random.default_rng(1).normal(size=(4, 3, 10_000)) + 0j
        out = ev(z)
        assert out.shape == (2, 3, 10_000)
        np.testing.assert_allclose(out[0], 2 * z[0] + z[3] ** 3)
        np.testing.assert_allclose(out[1], 1.0)

    def test_psi_inverse_round_trip(self, octupole_setup):
        nf = octupole_setup[2]
        tr = _Transforms(nf, "x")
        th = np.random.default_rng(2).uniform(0, 2 * np.pi, (2, 300, 1))
        zeta = tr.torus(np.full((300, 1), 5.0), np.full((300, 1), 1.5), th[0], th[1])
        z = tr.psi_inverse(zeta, 1e-12)
        assert np.all(np.isfinite(z))
        np.testing.assert_allclose(tr.psi(_to_complex(z)), zeta, atol=1e-10)

    def test_psi_inverse_linear_is_identity(self):
        tr = _Transforms(linear_normal_form(), "y")
        th = np.random.default_rng(3).uniform(0, 2 * np.pi, (2, 50, 1))
        zeta = tr.torus(np.full((50, 1), 4.0), np.full((50, 1), 2.0), th[0], th[1])
        np.testing.assert_allclose(tr.psi_inverse(zeta, 1e-12), _to_real(zeta), atol=1e-12)
        # the plane of the collimator is y: its amplitude is |zeta2|
        np.testing.assert_allclose(np.abs(zeta[2]), 4.0)


# --------------------------------------------------------------------------- #
# Linear limit: the pencil of xpart / xcoll
# --------------------------------------------------------------------------- #


class TestLinearPencil:
    def test_region(self, linear_setup):
        line, tw, _, sig = linear_setup
        p = pencil(linear_setup, 4000, side="+")
        n = tw.get_normalized_coordinates(p, nemitt_x=NEMITT, nemitt_y=NEMITT)
        cut, dr = JAW_SIGMAS, SPREAD_SIGMAS
        assert np.all(p.x >= JAW_SIGMAS * sig - 1e-15)
        assert np.all(n.x_norm >= cut - 1e-9)
        r = np.hypot(n.x_norm, n.px_norm)
        assert np.all(r <= cut + dr + 1e-6)
        # the pencil fills the region: radii up to cut + dr, angles both ways
        assert r.max() > cut + 0.99 * dr
        half_width = np.sqrt((cut + dr) ** 2 - cut**2)
        assert n.px_norm.min() < -0.9 * half_width and n.px_norm.max() > 0.9 * half_width

    def test_matches_xpart_pencil(self, linear_setup):
        xp = pytest.importorskip("xpart")
        tw = linear_setup[1]
        p = pencil(linear_setup, 20_000, side="+")
        n = tw.get_normalized_coordinates(p, nemitt_x=NEMITT, nemitt_y=NEMITT)
        np.random.seed(5)
        ref_x, ref_px, _, _ = xp.generate_2D_pencil(20_000, JAW_SIGMAS, SPREAD_SIGMAS, side="+")
        for ours, ref in ((n.x_norm, ref_x), (n.px_norm, ref_px)):
            err = 5 * np.std(ref) / np.sqrt(ref.size)
            assert abs(np.mean(ours) - np.mean(ref)) < err
            assert np.std(ours) == pytest.approx(np.std(ref), rel=0.03)
        # other plane: Gaussian in normalised coordinates
        assert np.std(n.y_norm) == pytest.approx(1.0, rel=0.03)
        assert np.std(n.py_norm) == pytest.approx(1.0, rel=0.03)

    def test_both_sides(self, linear_setup):
        p = pencil(linear_setup, 301)
        x = np.asarray(p.x)
        assert x.size == 301
        assert np.all(x[:150] > 0) and np.all(x[150:] < 0)

    def test_right_jaw_and_jaw_pair(self, linear_setup):
        sig = linear_setup[3]
        p = pencil(linear_setup, 200, side="-", jaw=(None, -4.0 * sig))
        assert np.all(p.x <= -4.0 * sig)
        n = linear_setup[1].get_normalized_coordinates(p, nemitt_x=NEMITT, nemitt_y=NEMITT)
        assert np.all(np.hypot(n.x_norm, n.px_norm) <= 4.0 + SPREAD_SIGMAS + 1e-6)

    def test_vertical_plane(self, linear_setup):
        line, tw, nf, _ = linear_setup
        sig_y = beam_size(tw, BETY)
        p = pencil(linear_setup, 300, plane="y", side="+", jaw=JAW_SIGMAS * sig_y,
                   pencil_spread=SPREAD_SIGMAS * sig_y)
        n = tw.get_normalized_coordinates(p, nemitt_x=NEMITT, nemitt_y=NEMITT)
        assert np.all(p.y >= JAW_SIGMAS * sig_y - 1e-15)
        assert np.all(np.hypot(n.y_norm, n.py_norm) <= JAW_SIGMAS + SPREAD_SIGMAS + 1e-6)

    def test_impact_parameter(self, linear_setup):
        sig = linear_setup[3]
        p = pencil(linear_setup, 200, side="+", impact_parameter=0.5 * sig)
        assert np.all(p.x >= (JAW_SIGMAS + 0.5) * sig - 1e-15)

    def test_seed_reproducible(self, linear_setup):
        a, b = pencil(linear_setup, 50, seed=7), pencil(linear_setup, 50, seed=7)
        np.testing.assert_array_equal(a.x, b.x)
        np.testing.assert_array_equal(a.py, b.py)

    def test_longitudinal_coordinates(self, linear_setup):
        zeta, delta = np.linspace(-1e-3, 1e-3, 40), np.zeros(40)
        p = pencil(linear_setup, 40, longitudinal=(zeta, delta))
        np.testing.assert_allclose(p.zeta, zeta, atol=1e-12)
        p = pencil(linear_setup, 40, longitudinal={"zeta": zeta, "delta": delta})
        np.testing.assert_allclose(p.zeta, zeta, atol=1e-12)
        with pytest.raises(NotImplementedError):
            pencil(linear_setup, 4, longitudinal="matched_dispersion")
        with pytest.raises(ValueError):
            pencil(linear_setup, 4, longitudinal="unknown")

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(plane="z"),
            dict(side="+-+"),
            dict(pencil_spread=0.0),
            dict(jaw=-1e-3, side="+"),  # left jaw below the closed orbit
            dict(jaw=(1e-3, None), side="-"),  # missing right jaw
            dict(nemitt_x=None),  # the element has no optics to read them from
        ],
    )
    def test_invalid_arguments(self, linear_setup, kwargs):
        with pytest.raises(ValueError):
            pencil(linear_setup, 10, **kwargs)

    def test_resonant_normal_form_rejected(self, linear_setup):
        nf = NormalForm4D(linear_normal_form().N, 2, "resonant", res_basis1=[4, 0])
        nf.compute_normal_form()
        with pytest.raises(ValueError, match="non-resonant"):
            pencil(linear_setup, 10, normal_form=nf)

    def test_normal_form_not_computed(self, linear_setup):
        nf = NormalForm4D(linear_normal_form().N, 2)
        with pytest.raises(ValueError, match="not been computed"):
            pencil(linear_setup, 10, normal_form=nf)


# --------------------------------------------------------------------------- #
# Non-linear pencil: orbits penetrate by at most pencil_spread
# --------------------------------------------------------------------------- #


class TestNonlinearPencil:
    @pytest.mark.parametrize("side", ["+", "-"])
    def test_orbit_depth_within_spread(self, octupole_setup, side):
        line, _, _, sig = octupole_setup
        spread = SPREAD_SIGMAS * sig
        p = pencil(octupole_setup, 500, side=side)
        depth, tracked = orbit_depth(line, p, JAW_SIGMAS * sig)
        assert np.all(tracked.state > 0)
        assert np.all(np.sign(p.x) * p.x >= JAW_SIGMAS * sig - 1e-15)
        assert depth.max() < 1.03 * spread
        # the whole range of depths is populated
        assert np.percentile(depth, 95) > 0.85 * spread
        assert np.percentile(depth, 5) < 0.3 * spread

    def test_linear_pencil_overshoots(self, octupole_setup):
        """The check above is meaningful: the linear pencil fails it."""
        line, tw, _, sig = octupole_setup
        p = pencil(octupole_setup, 500, side="+", normal_form=linear_normal_form(tw.qx % 1, tw.qy % 1))
        depth, _ = orbit_depth(line, p, JAW_SIGMAS * sig)
        assert depth.max() > 1.5 * SPREAD_SIGMAS * sig


# --------------------------------------------------------------------------- #
# LHC primary collimator
# --------------------------------------------------------------------------- #


def test_lhc_primary_collimator():
    if not os.path.exists(LHC_FILE):
        pytest.skip("LHC test line not available")
    name = "tcp.c6l7.b1"
    line = xt.Line.from_json(LHC_FILE)
    line.vars["i_mo_b1"] = -26
    line.build_tracker()
    # normal form at the collimator, on a copy of the line starting there
    cycled = line.copy()
    cycled.cycle(name_first_element=name)
    cycled.build_tracker()
    otm = OneTurnMap4D(cycled, 8, NEMITT, NEMITT, sigma_limit=8, seed=1)
    otm.compute()
    nf = otm.calculate_normal_form(7)

    tw = line.twiss()
    k = list(tw.name).index(name)
    sig = beam_size(tw, float(tw.betx[k]))
    jaw = float(tw.x[k]) + JAW_SIGMAS * sig
    spread = SPREAD_SIGMAS * sig
    p = generate_pencil_on_collimator(
        line, name, 100, normal_form=nf, plane="x", jaw=(jaw, None), side="+",
        nemitt_x=NEMITT, nemitt_y=NEMITT, pencil_spread=spread, twiss=tw, seed=2,
    )
    assert np.all(p.at_element == k)
    assert np.all(p.x >= jaw - 1e-15)
    # track on the cycled line, which starts at the collimator
    q = cycled.build_particles(x=p.x, px=p.px, y=p.y, py=p.py, zeta=p.zeta, delta=p.delta)
    depth, tracked = orbit_depth(cycled, q, jaw, num_turns=256)
    assert np.all(tracked.state > 0)
    assert depth.max() < 1.05 * spread


# --------------------------------------------------------------------------- #
# Distributions sampled in normal form coordinates
# --------------------------------------------------------------------------- #


def radius_drift(line, tw, z, num_turns: int) -> float:
    """Largest shift of the quantiles of hypot(x_norm, px_norm) after tracking."""
    p = line.build_particles(x_norm=z[0], px_norm=z[1], y_norm=z[2], py_norm=z[3],
                             nemitt_x=NEMITT, nemitt_y=NEMITT, zeta=0, delta=0)

    def radius():
        n = tw.get_normalized_coordinates(p, nemitt_x=NEMITT, nemitt_y=NEMITT)
        return np.hypot(n.x_norm, n.px_norm)

    q = np.linspace(0.02, 0.98, 25)
    before = np.quantile(radius(), q)
    line.track(p, num_turns=num_turns)
    assert np.all(p.state > 0)
    return float(np.max(np.abs(np.quantile(radius(), q) - before)))


class TestCircularSector:
    def test_linear_is_xpart(self):
        x, px, y, py, r, th = generate_2D_uniform_circular_sector(
            5000, (1.0, 2.0), (0.0, np.pi / 2), normal_form=linear_normal_form(), seed=0)
        np.testing.assert_allclose(x, r * np.cos(th), atol=1e-12)
        np.testing.assert_allclose(px, r * np.sin(th), atol=1e-12)
        assert np.all((r >= 1.0) & (r <= 2.0) & (th >= 0) & (th <= np.pi / 2))
        # uniform in area: r^2 uniform
        assert abs(np.mean((r**2 - 1) / 3) - 0.5) < 0.02
        assert abs(np.std(y) - 1) < 0.05 and abs(np.std(py) - 1) < 0.05

    def test_vertical_plane_and_no_other_plane(self):
        x, px, y, py, r, _ = generate_2D_uniform_circular_sector(
            100, (3.0, 4.0), normal_form=linear_normal_form(), plane="y", sigma_other=0.0, seed=0)
        np.testing.assert_allclose(np.hypot(y, py), r)
        assert np.all((r >= 3.0) & (r <= 4.0))
        assert np.all(x == 0) and np.all(px == 0)

    def test_seed(self):
        a = generate_2D_uniform_circular_sector(10, (1, 2), normal_form=linear_normal_form(), seed=4)
        b = generate_2D_uniform_circular_sector(10, (1, 2), normal_form=linear_normal_form(), seed=4)
        np.testing.assert_array_equal(np.array(a), np.array(b))

    def test_nonlinear_amplitudes(self, octupole_setup):
        nf = octupole_setup[2]
        x, px, y, py, r, th = generate_2D_uniform_circular_sector(
            2000, (4.0, 5.0), (-0.5, 0.5), normal_form=nf, seed=0)
        rho, _ = _Transforms(nf, "x").amplitudes(np.array([x, px, y, py]))
        np.testing.assert_allclose(rho, r, atol=1e-8)
        assert np.all((r >= 4.0 - 1e-8) & (r <= 5.0 + 1e-8))
        assert np.all((th >= -0.5 - 1e-8) & (th <= 0.5 + 1e-8))
        # not the linear ring (for which none would lie outside 4 <= hypot(x, px) <= 5)
        r_lin = np.hypot(x, px)
        assert np.mean((r_lin < 4.0) | (r_lin > 5.0)) > 0.01

    def test_nonlinear_ring_is_matched(self, octupole_setup):
        line, tw, nf, _ = octupole_setup
        lin = linear_normal_form(tw.qx % 1, tw.qy % 1)
        drift = {}
        for label, form in (("nonlinear", nf), ("linear", lin)):
            z = generate_2D_uniform_circular_sector(10_000, (4.0, 5.0), normal_form=form, seed=1)
            drift[label] = radius_drift(line, tw, z, 300)
        assert drift["nonlinear"] < 0.012
        assert drift["linear"] > 2.5 * drift["nonlinear"]

    @pytest.mark.parametrize("kwargs", [dict(plane="z"), dict(sigma_other=-1.0)])
    def test_invalid_arguments(self, kwargs):
        with pytest.raises(ValueError):
            generate_2D_uniform_circular_sector(10, normal_form=linear_normal_form(), **kwargs)

    def test_resonant_normal_form_warns(self):
        nf = NormalForm4D(linear_normal_form().N, 2, "resonant", res_basis1=[4, 0])
        nf.compute_normal_form()
        with pytest.warns(UserWarning, match="approximately matched"):
            out = generate_2D_uniform_circular_sector(10, normal_form=nf)
        assert out[0].shape == (10,)


class TestGaussianBunch:
    def bunch(self, setup, num_particles=5000, **kwargs):
        line, tw, nf, _ = setup
        args = dict(line=line, normal_form=nf, at_element="coll", twiss=tw, seed=0)
        args.update(kwargs)
        return generate_matched_gaussian_bunch(num_particles, NEMITT, NEMITT, 0.08, **args)

    def test_linear_is_gaussian(self, linear_setup):
        line, tw, _, _ = linear_setup
        p = self.bunch(linear_setup, total_intensity_particles=1e11)
        n = tw.get_normalized_coordinates(p, nemitt_x=NEMITT, nemitt_y=NEMITT)
        for u in (n.x_norm, n.px_norm, n.y_norm, n.py_norm):
            assert abs(np.mean(u)) < 0.05 and abs(np.std(u) - 1) < 0.05
        assert np.all(p.at_element == 0)
        np.testing.assert_allclose(p.weight, 1e11 / 5000)
        assert abs(np.std(p.zeta) - 0.08) < 0.01

    def test_nonlinear_normal_form_coordinates_are_gaussian(self, octupole_setup):
        line, tw, nf, _ = octupole_setup
        p = self.bunch(octupole_setup)
        n = tw.get_normalized_coordinates(p, nemitt_x=NEMITT, nemitt_y=NEMITT)
        # the transverse coordinates are built at delta != 0 through the W matrix;
        # the toy line has no dispersion, so the normalised coordinates are exact
        w = _to_real(_Transforms(nf, "x").psi(_to_complex(
            np.array([n.x_norm, n.px_norm, n.y_norm, n.py_norm]))))
        np.testing.assert_allclose(np.std(w, axis=1), 1.0, atol=0.05)

    def test_seed_and_matcher(self, linear_setup):
        out = self.bunch(linear_setup, 10, return_matcher=True, seed=3)
        assert isinstance(out, tuple) and len(out) == 2
        q = self.bunch(linear_setup, 10, seed=3)
        np.testing.assert_array_equal(out[0].x, q.x)

    def test_tracker_required(self, linear_setup):
        line, tw, nf, _ = linear_setup
        line2 = xt.Line(elements=[xt.Drift(length=1.0)])
        with pytest.raises(ValueError):
            generate_matched_gaussian_bunch(10, NEMITT, NEMITT, 0.08, line=line2,
                                            normal_form=nf, at_element="coll", twiss=tw)


# --------------------------------------------------------------------------- #
# q-Gaussian helpers
# --------------------------------------------------------------------------- #


def q_gaussian_1d_cdf(q: float, beta: float):
    """CDF of the 1D q-Gaussian e_q(-beta x^2) (a Student t), Gaussian for q = 1."""
    stats = pytest.importorskip("scipy.stats")
    if q == 1:
        return stats.norm(scale=np.sqrt(1 / (2 * beta))).cdf
    nu, s = (3 - q) / (q - 1), 1 / np.sqrt((3 - q) * beta)
    return lambda x: stats.t.cdf(x / s, nu)


def plane_radius_cdf(q: float, beta: float):
    """CDF of the radius of the 2D q-Gaussian of _f_2D."""
    if q == 1:
        return lambda r: 1 - np.exp(-beta * r**2)
    return lambda r: 1 - (1 + beta * (q - 1) * r**2) ** (-(3 - q) / (2 * (q - 1)))


GENERATORS = [_generate_factorizable, _generate_nonfactorizable]


class TestQGaussianHelpers:
    @pytest.mark.parametrize("q", [1.2, 2.0, 2.8])
    def test_densities_normalised(self, q):
        from scipy.integrate import quad

        beta = 0.7
        n2 = quad(lambda r: 2 * np.pi * r * _f_2D(r, q, beta), 0, np.inf)[0]
        n4 = quad(lambda m: 2 * np.pi**2 * m**3 * _f_4D(m, q, beta), 0, np.inf)[0]
        assert n2 == pytest.approx(1, abs=1e-6)
        assert n4 == pytest.approx(1, abs=1e-6)

    @pytest.mark.parametrize("generate", GENERATORS)
    @pytest.mark.parametrize("q", [1.0, 1.5, 2.5])
    def test_marginals(self, generate, q):
        from scipy.stats import kstest

        beta = 0.8
        w = generate(50_000, q, beta, np.random.default_rng(0))
        assert w.shape == (4, 50_000)
        for u in w:
            assert kstest(u, q_gaussian_1d_cdf(q, beta)).statistic < 0.012
        for r in (np.hypot(w[0], w[1]), np.hypot(w[2], w[3])):
            assert kstest(r, plane_radius_cdf(q, beta)).statistic < 0.012

    def test_unit_variance(self):
        q = 1.2
        for generate in GENERATORS:
            w = generate(100_000, q, 1 / (5 - 3 * q), np.random.default_rng(1))
            np.testing.assert_allclose(np.std(w, axis=1), 1.0, atol=0.02)

    def test_planes_independent_only_if_factorisable(self):
        from scipy.stats import spearmanr

        rng = np.random.default_rng(2)
        corr = {}
        for generate in GENERATORS:
            w = generate(50_000, 2.0, 1.0, rng)
            corr[generate] = spearmanr(np.hypot(w[0], w[1]), np.hypot(w[2], w[3])).statistic
        assert abs(corr[_generate_factorizable]) < 0.02
        assert corr[_generate_nonfactorizable] > 0.5

    def test_seed(self):
        for generate in GENERATORS:
            a = generate(10, 1.5, 1.0, np.random.default_rng(3))
            b = generate(10, 1.5, 1.0, np.random.default_rng(3))
            np.testing.assert_array_equal(a, b)

    @pytest.mark.parametrize("q, beta", [(0.5, 1.0), (3.0, 1.0), (1.5, 0.0)])
    def test_invalid_parameters(self, q, beta):
        for generate in GENERATORS:
            with pytest.raises(ValueError):
                generate(10, q, beta, np.random.default_rng(0))


# --------------------------------------------------------------------------- #
# q-Gaussian bunch
# --------------------------------------------------------------------------- #


def qgaussian(setup, num_particles=4000, **kwargs):
    line, tw, nf, _ = setup
    args = dict(line=line, normal_form=nf, at_element="coll", twiss=tw, seed=0,
                q=1.5, beta=1.0, fact=True)
    args.update(kwargs)
    p = generate_matched_qgaussian_bunch(num_particles, NEMITT, NEMITT, 0.08, **args)
    n = tw.get_normalized_coordinates(p, nemitt_x=NEMITT, nemitt_y=NEMITT)
    return p, np.array([n.x_norm, n.px_norm, n.y_norm, n.py_norm])


class TestQGaussianBunch:
    @pytest.mark.parametrize("fact", [True, False])
    def test_linear_marginals(self, linear_setup, fact):
        from scipy.stats import kstest

        p, z = qgaussian(linear_setup, 20_000, fact=fact, q=1.4, beta=0.8)
        assert len(p.x) == 20_000 and np.all(p.at_element == 0)
        for u in z:
            assert kstest(u, q_gaussian_1d_cdf(1.4, 0.8)).statistic < 0.015

    @pytest.mark.parametrize("skew_cut", [True, False])
    @pytest.mark.parametrize("fact", [True, False])
    def test_cut(self, linear_setup, skew_cut, fact):
        p, z = qgaussian(linear_setup, 3000, q=2.0, sigma_cut_x=3.0, sigma_cut_y=2.0,
                         skew_cut=skew_cut, fact=fact)
        rx, ry = np.hypot(z[0], z[1]), np.hypot(z[2], z[3])
        assert len(p.x) == 3000
        if skew_cut:
            assert np.all(rx**2 / 9 + ry**2 / 4 < 1 + 1e-9)
            # the corners of the rectangle are cut
            assert not np.any((rx > 2.4) & (ry > 1.6))
        else:
            assert np.all((rx < 3 + 1e-9) & (ry < 2 + 1e-9))
            assert np.any((rx > 2.4) & (ry > 1.6))
        assert rx.max() > 2.8 and ry.max() > 1.8

    @pytest.mark.parametrize("limits", [(3.0, 2.0), (2.0, None), (None, 2.5)])
    @pytest.mark.parametrize("skew_cut", [True, False])
    def test_torus_cut_is_normalised_cut_if_linear(self, limits, skew_cut):
        nf = linear_normal_form()
        cut, nf_cut = _cuts([nf], [_Transforms(nf, "x")], np.empty(0), *limits, skew_cut, False)
        assert cut is None
        w = np.random.default_rng(0).normal(size=(4, 100_000)) * 1.5
        np.testing.assert_array_equal(nf_cut(w)[0], _normalised_cut(*limits, skew_cut)(w, 1.0))

    @pytest.mark.parametrize("skew_cut", [True, False])
    def test_cut_tori_stay_within_cut(self, octupole_setup, skew_cut):
        line, tw = octupole_setup[:2]
        sx, sy = 5.0, 3.0
        p, _ = qgaussian(octupole_setup, 2000, q=1.5, beta=0.5, sigma_cut_x=sx,
                         sigma_cut_y=sy, skew_cut=skew_cut)
        worst = 0.0
        for _ in range(200):
            n = tw.get_normalized_coordinates(p, nemitt_x=NEMITT, nemitt_y=NEMITT)
            a, b = np.hypot(n.x_norm, n.px_norm) / sx, np.hypot(n.y_norm, n.py_norm) / sy
            worst = max(worst, float(np.max(np.hypot(a, b) if skew_cut else np.maximum(a, b))))
            line.track(p, num_turns=1)
        # whole tori are within the cut, up to the accuracy of the normal form
        # (the cut of the particles in normalised coordinates leaks ~1% by 1-10%)
        assert 0.99 < worst < 1.005

    def test_cut_in_one_plane(self, linear_setup):
        _, z = qgaussian(linear_setup, 3000, q=2.0, sigma_cut_x=2.0)
        rx, ry = np.hypot(z[0], z[1]), np.hypot(z[2], z[3])
        assert np.all(rx < 2 + 1e-9) and ry.max() > 5

    def test_nonlinear_normal_form_coordinates(self, octupole_setup):
        from scipy.stats import kstest

        nf = octupole_setup[2]
        q, beta = 1.2, 1 / (5 - 3 * 1.2)
        _, z = qgaussian(octupole_setup, 5000, q=q, beta=beta, sigma_cut_x=6.0, sigma_cut_y=6.0)
        assert np.all(np.hypot(z[0], z[1]) ** 2 + np.hypot(z[2], z[3]) ** 2 < 36 + 1e-9)
        w = _to_real(_Transforms(nf, "x").psi(_to_complex(z)))
        # the cut at 6 sigma removes ~1e-3 of the q-Gaussian
        assert kstest(np.hypot(w[0], w[1]), plane_radius_cdf(q, beta)).statistic < 0.03

    def test_seed(self, linear_setup):
        a, _ = qgaussian(linear_setup, 10, seed=5, sigma_cut_x=3.0)
        b, _ = qgaussian(linear_setup, 10, seed=5, sigma_cut_x=3.0)
        np.testing.assert_array_equal(a.x, b.x)

    def test_cut_too_tight(self, linear_setup):
        with pytest.raises(RuntimeError, match="too tight"):
            qgaussian(linear_setup, 10, q=2.5, sigma_cut_x=1e-4, sigma_cut_y=1e-4)

    @pytest.mark.parametrize(
        "kwargs, error",
        [
            (dict(q=3.0), ValueError),
            (dict(q=0.9), ValueError),
            (dict(beta=0.0), ValueError),
            (dict(sigma_cut_x=-1.0), ValueError),
            (dict(sigma_cut_y=0.0), ValueError),
            (dict(fact=1), TypeError),
            (dict(skew_cut="yes"), TypeError),
            (dict(strict_cut=1), TypeError),
        ],
    )
    def test_invalid_arguments(self, linear_setup, kwargs, error):
        with pytest.raises(error):
            qgaussian(linear_setup, 10, **kwargs)


# --------------------------------------------------------------------------- #
# Resonant normal forms in the distributions
# --------------------------------------------------------------------------- #


def coupling_normal_form(qx: float = QX, qy: float = QY) -> NormalForm4D:
    """Quasi-resonant normal form of the coupling resonance of a rotation."""
    q = 0.5 * (qx + qy)
    nf = NormalForm4D(linear_normal_form(qx, qy).N, 2, "quasiresonant",
                      res_eig=[np.exp(2j * np.pi * q), np.exp(-2j * np.pi * q)] * 2,
                      res_basis1=[1, -1])
    nf.compute_normal_form()
    return nf


class TestResonantNormalForms:
    def _bunch(self, setup, function, **kwargs):
        line, tw, _, _ = setup
        return function(200, NEMITT, NEMITT, 0.08, line=line, at_element="coll", twiss=tw,
                        seed=0, **kwargs)

    def test_gaussian_bunch_with_coupling_resonance(self, linear_setup):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            p = self._bunch(linear_setup, generate_matched_gaussian_bunch,
                            normal_form=coupling_normal_form())
        assert len(p.x) == 200

    def test_qgaussian_with_coupling_resonance(self, linear_setup):
        nf = coupling_normal_form()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self._bunch(linear_setup, generate_matched_qgaussian_bunch, normal_form=nf,
                        q=1.5, beta=1.0, fact=False)
            # q = 1 is the Gaussian, a function of the sum of the actions
            self._bunch(linear_setup, generate_matched_qgaussian_bunch, normal_form=nf,
                        q=1.0, beta=0.5, fact=True)
        with pytest.warns(UserWarning, match="approximately matched"):
            self._bunch(linear_setup, generate_matched_qgaussian_bunch, normal_form=nf,
                        q=1.5, beta=1.0, fact=True)

    def test_cut_with_coupling_resonance(self):
        nf = coupling_normal_form()
        args = ([nf], [_Transforms(nf, "x")], np.empty(0))
        w = np.random.default_rng(1).normal(size=(4, 100_000)) * 1.5
        # elliptical with equal limits: the shell cut, the same as the linear cut
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cut, nf_cut = _cuts(*args, 2.5, 2.5, True, False)
        assert cut is None
        np.testing.assert_array_equal(nf_cut(w)[0], _normalised_cut(2.5, 2.5, True)(w, 1.0))
        # otherwise the cut in normalised coordinates by default ...
        for limits, skew in (((3.0, 2.0), True), ((2.5, 2.5), False), ((2.5, None), True)):
            with pytest.warns(UserWarning, match="normalised coordinates instead"):
                cut, nf_cut = _cuts(*args, *limits, skew, False)
            assert nf_cut is None
            np.testing.assert_array_equal(cut(w, 1.0), _normalised_cut(*limits, skew)(w, 1.0))
        # ... or the stricter shell cut
        with pytest.warns(UserWarning, match="removes tori within the cut"):
            cut, nf_cut = _cuts(*args, 3.0, 2.0, False, True)
        assert cut is None
        np.testing.assert_array_equal(nf_cut(w)[0], np.sum(w**2, axis=0) < 4.0)

    def test_gaussian_with_other_resonance_warns(self, linear_setup):
        nf = NormalForm4D(linear_normal_form().N, 2, "resonant", res_basis1=[4, 0])
        nf.compute_normal_form()
        with pytest.warns(UserWarning, match="approximately matched"):
            self._bunch(linear_setup, generate_matched_gaussian_bunch, normal_form=nf)

    def test_double_resonance_rejected(self, linear_setup):
        nf = NormalForm4D(linear_normal_form().N, 2, "resonant", res_basis1=[1, -1],
                          res_basis2=[4, 0])
        nf.compute_normal_form()
        with pytest.raises(ValueError, match="two resonances"):
            self._bunch(linear_setup, generate_matched_gaussian_bunch, normal_form=nf)

    def test_pencil_still_needs_nonresonant(self, linear_setup):
        with pytest.raises(ValueError, match="non-resonant"):
            pencil(linear_setup, 10, normal_form=coupling_normal_form())


# --------------------------------------------------------------------------- #
# Seams between normal forms
# --------------------------------------------------------------------------- #


class TestSeamMismatch:
    def test_same_normal_form_has_no_mismatch(self, octupole_setup):
        nf = octupole_setup[2]
        sm = seam_mismatch(nf, nf, [2.0, 3.0], num_points=300, seed=0)
        assert sm.radii.tolist() == [2.0, 3.0]
        for d in sm.inner_to_outer + sm.outer_to_inner:
            assert d.size == 300
            np.testing.assert_allclose(d, 0, atol=1e-9)
        assert np.all(sm.failed == 0)

    def test_different_normal_forms(self, octupole_setup):
        tw = octupole_setup[1]
        lin = linear_normal_form(tw.qx % 1, tw.qy % 1)
        sm = seam_mismatch(lin, octupole_setup[2], [1.0, 4.0], num_points=300, seed=0)
        rms = sm.summary()["rms"]
        assert 0 < rms[0] < rms[1]  # the non-linear distortion grows with amplitude
        assert sm.best_radius == 1.0
        summary = sm.summary()
        for key in ("inner_to_outer_mean", "inner_to_outer_rms", "inner_to_outer_max",
                    "outer_to_inner_mean", "outer_to_inner_rms", "outer_to_inner_max", "rms"):
            assert summary[key].shape == (2,)

    def test_coupling_resonance_accepted(self):
        nf = coupling_normal_form()
        sm = seam_mismatch(nf, linear_normal_form(), 2.0, num_points=50, seed=0)
        np.testing.assert_allclose(sm.inner_to_outer[0], 0, atol=1e-9)

    def test_other_resonance_rejected(self):
        nf = NormalForm4D(linear_normal_form().N, 2, "resonant", res_basis1=[4, 0])
        nf.compute_normal_form()
        with pytest.raises(ValueError, match="not invariant"):
            seam_mismatch(nf, linear_normal_form(), 2.0)

    def test_invalid(self):
        nf = linear_normal_form()
        with pytest.raises(ValueError, match="radii"):
            seam_mismatch(nf, nf, [-1.0])
        with pytest.raises(ValueError, match="num_points"):
            seam_mismatch(nf, nf, 1.0, num_points=0)
        with pytest.raises(ValueError, match="not been computed"):
            seam_mismatch(NormalForm4D(nf.N, 2), nf, 1.0)


# --------------------------------------------------------------------------- #
# Stitched normal forms
# --------------------------------------------------------------------------- #


class TestStitchedBunch:
    def _normalised(self, setup, function, **kwargs):
        line, tw, _, _ = setup
        p = function(3000, NEMITT, NEMITT, 0.08, line=line, at_element="coll", twiss=tw,
                     seed=4, **kwargs)
        n = tw.get_normalized_coordinates(p, nemitt_x=NEMITT, nemitt_y=NEMITT)
        return np.array([n.x_norm, n.px_norm, n.y_norm, n.py_norm])

    def test_same_normal_form_is_unchanged(self, octupole_setup):
        nf = octupole_setup[2]
        single = self._normalised(octupole_setup, generate_matched_gaussian_bunch, normal_form=nf)
        stitched = self._normalised(octupole_setup, generate_matched_gaussian_bunch,
                                    normal_form=[nf, nf, nf], seams=[1.0, 2.0])
        np.testing.assert_array_equal(single, stitched)

    @pytest.mark.parametrize("function, kwargs", [
        (generate_matched_gaussian_bunch, {}),
        (generate_matched_qgaussian_bunch, dict(q=1.5, beta=1.0, fact=False, sigma_cut_x=6.0)),
    ])
    def test_points_follow_their_normal_form(self, octupole_setup, function, kwargs):
        tw, nf = octupole_setup[1], octupole_setup[2]
        lin = linear_normal_form(tw.qx % 1, tw.qy % 1)
        seam = 2.0
        z = self._normalised(octupole_setup, function, normal_form=[lin, nf], seams=[seam], **kwargs)
        rho_in = np.sqrt(np.sum(z**2, axis=0))  # linear normal form: Psi = identity
        w = _to_real(_Transforms(nf, "x").psi(_to_complex(z)))
        rho_out = np.sqrt(np.sum(w**2, axis=0))
        inner, outer = rho_in < seam - 1e-6, rho_out >= seam - 1e-6
        # every point comes from the normal form of its side of the seam
        assert np.all(inner | outer)
        assert inner.sum() > 100 and (~inner).sum() > 100

    @pytest.mark.parametrize("kwargs, match", [
        (dict(seams=[2.0]), "sequence"),
        (dict(normal_form="pair", seams=[2.0, 3.0]), "seam radii"),
        (dict(normal_form="pair", seams=None), "seam radii"),
        (dict(normal_form="pair", seams=[-1.0]), "positive and increasing"),
        (dict(normal_form="five", seams=[1.0, 2.0, 3.0, 4.0]), "Between 2 and 4"),
    ])
    def test_invalid_seams(self, octupole_setup, kwargs, match):
        nf = octupole_setup[2]
        forms = {"pair": [nf, nf], "five": [nf] * 5}
        kwargs = dict(kwargs)
        kwargs["normal_form"] = forms.get(kwargs.get("normal_form"), nf)
        with pytest.raises(ValueError, match=match):
            self._normalised(octupole_setup, generate_matched_gaussian_bunch, **kwargs)

    def test_decreasing_seams_rejected(self, octupole_setup):
        nf = octupole_setup[2]
        with pytest.raises(ValueError, match="increasing"):
            self._normalised(octupole_setup, generate_matched_gaussian_bunch,
                             normal_form=[nf, nf, nf], seams=[3.0, 2.0])

    def test_factorisable_cannot_be_stitched(self, octupole_setup):
        nf = octupole_setup[2]
        with pytest.raises(ValueError, match="joined"):
            self._normalised(octupole_setup, generate_matched_qgaussian_bunch,
                             normal_form=[nf, nf], seams=[2.0], q=1.5, beta=1.0, fact=True)

    def test_resonance_without_common_invariant_rejected(self, octupole_setup):
        nf4 = NormalForm4D(linear_normal_form().N, 2, "resonant", res_basis1=[4, 0])
        nf4.compute_normal_form()
        with pytest.raises(ValueError, match="not invariant"):
            self._normalised(octupole_setup, generate_matched_gaussian_bunch,
                             normal_form=[octupole_setup[2], nf4], seams=[2.0])

    def test_coupling_normal_forms_can_be_stitched(self, linear_setup):
        nf = coupling_normal_form()
        z = self._normalised(linear_setup, generate_matched_gaussian_bunch,
                             normal_form=[nf, linear_normal_form()], seams=[1.5])
        assert z.shape == (4, 3000)
