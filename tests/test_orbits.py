from __future__ import annotations

import inspect
import os

import numpy as np
import pytest

from xnf4d.orbits import DEFAULT_ANGLES, Orbits, initial_conditions, track_orbits

NEMITT = 2.5e-6
LHC_FILE = os.path.join(
    os.path.dirname(__file__), "..", "test_data", "ramp.0_lhc_b1.json.gz"
)


# --------------------------------------------------------------------------- #
# Initial conditions
# --------------------------------------------------------------------------- #


class TestInitialConditions:
    def test_default_angles(self):
        assert DEFAULT_ANGLES == (0, 18, 36, 54, 72, 90)

    @pytest.mark.parametrize("plane", ["x", "y"])
    def test_single_plane(self, plane):
        planes, radius, angle, z = initial_conditions(plane, sigma_limit=5, radial_step=1)
        assert z.shape == (4, 30)
        assert np.all(planes == plane)
        np.testing.assert_allclose(np.unique(radius), [1, 2, 3, 4, 5])
        np.testing.assert_allclose(np.unique(angle), DEFAULT_ANGLES)
        i, j = (0, 1) if plane == "x" else (2, 3)
        other = [k for k in range(4) if k not in (i, j)]
        theta = np.deg2rad(angle)
        np.testing.assert_allclose(z[i], radius * np.cos(theta), atol=1e-15)
        np.testing.assert_allclose(z[j], radius * np.sin(theta), atol=1e-15)
        assert np.all(z[other] == 0)
        np.testing.assert_allclose(np.hypot(z[i], z[j]), radius)

    def test_both_planes(self):
        planes, radius, angle, z = initial_conditions("both", sigma_limit=3, radial_step=1)
        assert z.shape == (4, 36)
        assert list(planes[:18]) == ["x"] * 18 and list(planes[18:]) == ["y"] * 18
        # each plane: for each angle the radii in increasing order
        assert list(radius[:3]) == [1, 2, 3] and list(angle[:3]) == [0, 0, 0]
        assert np.all(z[2:, :18] == 0) and np.all(z[:2, 18:] == 0)

    def test_radii(self):
        _, radius, _, _ = initial_conditions("x", sigma_limit=1.0, radial_step=0.1)
        np.testing.assert_allclose(np.unique(radius), np.arange(1, 11) * 0.1)
        _, radius, _, _ = initial_conditions("x", sigma_limit=2.5, radial_step=1)
        np.testing.assert_allclose(np.unique(radius), [1, 2])
        assert radius.dtype == float

    def test_custom_angles(self):
        _, _, angle, z = initial_conditions("x", sigma_limit=1, radial_step=1, angles=[45])
        assert angle.tolist() == [45]
        np.testing.assert_allclose(z[:2, 0], [np.sqrt(0.5), np.sqrt(0.5)])

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            (dict(plane="z"), "plane"),
            (dict(radial_step=0), "radial_step"),
            (dict(sigma_limit=0.5, radial_step=1), "sigma_limit"),
            (dict(angles=[]), "angle"),
        ],
    )
    def test_invalid(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            initial_conditions(**kwargs)


# --------------------------------------------------------------------------- #
# Orbits container
# --------------------------------------------------------------------------- #


def synthetic_orbits(with_nf: bool) -> Orbits:
    planes, radius, angle, z = initial_conditions("both", sigma_limit=2, radial_step=1)
    n, t = len(planes), 5
    arr = lambda k: np.full((n, t), float(k))  # noqa: E731
    nf = {f"{c}_nf": arr(8 + i) for i, c in enumerate(("x", "px", "y", "py"))} if with_nf else {}
    return Orbits(
        plane=planes, radius=radius, angle=angle,
        lost=np.zeros(n, bool), num_valid_turns=np.full(n, t),
        x=arr(0), px=arr(1), y=arr(2), py=arr(3),
        x_norm=arr(4), px_norm=arr(5), y_norm=arr(6), py_norm=arr(7), **nf,
    )


class TestOrbitsContainer:
    def test_properties(self):
        o = synthetic_orbits(False)
        assert o.num_particles == 24 and o.num_turns == 5
        assert not o.has_normal_form
        assert synthetic_orbits(True).has_normal_form

    @pytest.mark.parametrize("with_nf", [False, True])
    def test_select(self, with_nf):
        o = synthetic_orbits(with_nf)
        sx = o.select("x")
        assert sx.num_particles == 12 and np.all(sx.plane == "x")
        assert sx.x_norm.shape == (12, 5)
        assert (sx.x_nf is None) == (not with_nf)
        np.testing.assert_array_equal(sx.radius, o.radius[o.plane == "x"])

    def test_select_invalid(self):
        with pytest.raises(ValueError):
            synthetic_orbits(False).select("both")


# --------------------------------------------------------------------------- #
# Tracking the LHC test line
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def lhc_line():
    xt = pytest.importorskip("xtrack")
    if not os.path.exists(LHC_FILE):
        pytest.skip("LHC test line not available")
    line = xt.Line.from_json(LHC_FILE)
    line.build_tracker()
    return line


@pytest.fixture(scope="module")
def twiss(lhc_line):
    return lhc_line.twiss()


@pytest.fixture(scope="module")
def normal_form(lhc_line):
    from xnf4d.one_turn_map import OneTurnMap4D

    otm = OneTurnMap4D(lhc_line, 6, nemitt_x=NEMITT, nemitt_y=NEMITT, seed=4)
    return otm.calculate_normal_form(5)


@pytest.fixture(scope="module")
def orbits(lhc_line, twiss, normal_form):
    return track_orbits(
        lhc_line, NEMITT, NEMITT, plane="both", sigma_limit=6, radial_step=2,
        num_turns=60, normal_form=normal_form, twiss=twiss,
    )


class TestTrackOrbits:
    def test_shapes_and_labels(self, orbits):
        assert orbits.num_particles == 36 and orbits.num_turns == 60
        for name in ("x", "px", "y", "py", "x_norm", "px_norm", "y_norm", "py_norm",
                     "x_nf", "px_nf", "y_nf", "py_nf"):
            assert getattr(orbits, name).shape == (36, 60), name
        planes, radius, angle, _ = initial_conditions("both", 6, 2)
        np.testing.assert_array_equal(orbits.plane, planes)
        np.testing.assert_array_equal(orbits.radius, radius)
        np.testing.assert_array_equal(orbits.angle, angle)
        assert not orbits.lost.any()
        assert np.all(orbits.num_valid_turns == 60)
        assert np.all(np.isfinite(orbits.x))

    def test_initial_normalised_coordinates(self, orbits):
        _, _, _, z = initial_conditions("both", 6, 2)
        got = np.array([orbits.x_norm[:, 0], orbits.px_norm[:, 0],
                        orbits.y_norm[:, 0], orbits.py_norm[:, 0]])
        np.testing.assert_allclose(got, z, atol=1e-10)

    def test_one_turn_matches_direct_tracking(self, lhc_line, twiss, orbits):
        z0 = [orbits.x_norm[:, 0], orbits.px_norm[:, 0], orbits.y_norm[:, 0], orbits.py_norm[:, 0]]
        p = lhc_line.build_particles(
            x_norm=z0[0], px_norm=z0[1], y_norm=z0[2], py_norm=z0[3],
            zeta_norm=0, pzeta_norm=0, nemitt_x=NEMITT, nemitt_y=NEMITT,
            W_matrix=twiss.W_matrix[0], particle_on_co=twiss.particle_on_co,
        )
        lhc_line.track(p, num_turns=1)
        tab = twiss.get_normalized_coordinates(p, nemitt_x=NEMITT, nemitt_y=NEMITT)
        order = np.argsort(np.asarray(tab.particle_id))
        for c in ("x", "px", "y", "py"):
            np.testing.assert_allclose(
                np.asarray(getattr(tab, f"{c}_norm"))[order],
                getattr(orbits, f"{c}_norm")[:, 1],
                atol=1e-12,
            )
            np.testing.assert_allclose(np.asarray(getattr(p, c))[order], getattr(orbits, c)[:, 1], atol=1e-15)

    def test_normal_form_actions_are_invariant(self, orbits):
        for plane, (a, b) in (("x", ("x", "px")), ("y", ("y", "py"))):
            o = orbits.select(plane)
            j_nf = (getattr(o, f"{a}_nf") ** 2 + getattr(o, f"{b}_nf") ** 2) / 2
            j_cs = (getattr(o, f"{a}_norm") ** 2 + getattr(o, f"{b}_norm") ** 2) / 2
            spread_nf = np.ptp(j_nf, axis=1) / j_nf.mean(axis=1)
            spread_cs = np.ptp(j_cs, axis=1) / j_cs.mean(axis=1)
            assert np.all(spread_nf < 1e-5)
            assert np.all(spread_cs > 100 * spread_nf)

    def test_without_normal_form(self, lhc_line, twiss):
        o = track_orbits(lhc_line, NEMITT, NEMITT, plane="y", sigma_limit=2,
                         radial_step=1, num_turns=3, twiss=twiss)
        assert o.x_nf is None and not o.has_normal_form
        assert o.num_particles == 12 and np.all(o.plane == "y")

    def test_lost_particles(self, lhc_line, twiss, normal_form):
        o = track_orbits(lhc_line, NEMITT, NEMITT, plane="x", sigma_limit=40,
                         radial_step=4, num_turns=40, normal_form=normal_form, twiss=twiss)
        assert o.lost.any() and not o.lost.all()
        for name in ("x", "py", "x_norm", "py_norm", "x_nf", "py_nf"):
            arr = getattr(o, name)
            for i, n in enumerate(o.num_valid_turns):
                if name in ("x", "py", "x_norm", "py_norm"):
                    assert np.all(np.isfinite(arr[i, :n])), (name, i)
                assert np.all(np.isnan(arr[i, n:])), (name, i)
        assert np.all(o.num_valid_turns[~o.lost] == 40)
        assert np.all(o.num_valid_turns[o.lost] < 40)
        # agreement with the turn at which xtrack reports the loss
        p = lhc_line.build_particles(
            x_norm=o.x_norm[:, 0], px_norm=o.px_norm[:, 0], y_norm=0, py_norm=0,
            zeta_norm=0, pzeta_norm=0, nemitt_x=NEMITT, nemitt_y=NEMITT,
            W_matrix=twiss.W_matrix[0], particle_on_co=twiss.particle_on_co,
        )
        lhc_line.track(p, num_turns=40)
        order = np.argsort(np.asarray(p.particle_id))
        state, at_turn = np.asarray(p.state)[order], np.asarray(p.at_turn)[order]
        np.testing.assert_array_equal(o.lost, state <= 0)
        np.testing.assert_array_equal(o.num_valid_turns[o.lost], at_turn[o.lost] + 1)

    def test_progress_bar_and_monitor(self, lhc_line, twiss, monkeypatch):
        calls = []
        original = lhc_line.track

        def spy(particles, **kwargs):
            calls.append(kwargs)
            return original(particles, **kwargs)

        monkeypatch.setattr(lhc_line, "track", spy)
        track_orbits(lhc_line, NEMITT, NEMITT, plane="x", sigma_limit=1,
                     radial_step=1, num_turns=4, twiss=twiss)
        assert calls == [dict(num_turns=4, turn_by_turn_monitor=True, with_progress=10)]

    def test_default_arguments(self):
        sig = inspect.signature(track_orbits).parameters
        assert sig["num_turns"].default == 2048
        assert sig["multithreaded"].default is False
        assert sig["plane"].default == "both"
        assert sig["normal_form"].default is None

    def test_twiss_computed_if_not_given(self, lhc_line, twiss):
        a = track_orbits(lhc_line, NEMITT, NEMITT, plane="x", sigma_limit=1,
                         radial_step=1, num_turns=3)
        b = track_orbits(lhc_line, NEMITT, NEMITT, plane="x", sigma_limit=1,
                         radial_step=1, num_turns=3, twiss=twiss)
        np.testing.assert_allclose(a.x_norm, b.x_norm, atol=1e-12)

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            (dict(num_turns=0), "num_turns"),
            (dict(method="5d"), "method"),
            (dict(nemitt_x=-1.0), "emittances"),
            (dict(plane="z"), "plane"),
        ],
    )
    def test_invalid_arguments(self, lhc_line, twiss, kwargs, match):
        args = dict(nemitt_x=NEMITT, nemitt_y=NEMITT, sigma_limit=1, radial_step=1,
                    num_turns=2, twiss=twiss)
        args.update(kwargs)
        with pytest.raises(ValueError, match=match):
            track_orbits(lhc_line, **args)

    def test_normal_form_not_computed(self, lhc_line, twiss):
        from xnf4d.numerical_normal_form import NormalForm4D
        from xnf4d.polynom import Map, Polynom, Term

        identity = Map(*(Polynom([Term(1.0, *(int(i == j) for i in range(4)))]) for j in range(4)))
        with pytest.raises(ValueError, match="not been computed"):
            track_orbits(lhc_line, NEMITT, NEMITT, sigma_limit=1, radial_step=1,
                         num_turns=2, twiss=twiss, normal_form=NormalForm4D(identity, 3))


class TestMultithreaded:
    def test_same_result_and_line_restored(self, lhc_line, twiss):
        kwargs = dict(plane="x", sigma_limit=4, radial_step=2, num_turns=20, twiss=twiss)
        serial = track_orbits(lhc_line, NEMITT, NEMITT, **kwargs)
        context = lhc_line._context
        threaded = track_orbits(lhc_line, NEMITT, NEMITT, multithreaded=True, **kwargs)
        np.testing.assert_array_equal(threaded.x, serial.x)
        np.testing.assert_array_equal(threaded.py_norm, serial.py_norm)
        # the line is back on a serial context and still tracks
        assert lhc_line._context.omp_num_threads == context.omp_num_threads == 0
        again = track_orbits(lhc_line, NEMITT, NEMITT, **kwargs)
        np.testing.assert_array_equal(again.x, serial.x)


# --------------------------------------------------------------------------- #
# Orbits from the interpolating Hamiltonian
# --------------------------------------------------------------------------- #

from xnf4d.numerical_normal_form import complex_to_real, real_to_complex  # noqa: E402
from xnf4d.orbits import (  # noqa: E402
    HamiltonianOrbits,
    _distance_to_polylines,
    _initial_planes,
    hamiltonian_orbits,
)
from xnf4d.polynom import Map, Polynom, Term  # noqa: E402


def _identity_map() -> Map:
    return Map(*(Polynom([Term(1.0, *(int(i == j) for i in range(4)))]) for j in range(4)))


class StubNormalForm:
    """Normal form with identity transformations and a given Hamiltonian."""

    def __init__(self, H: Polynom, tunes: tuple[float, float] = (0.31, 0.32)) -> None:
        self.H = H
        self.Psi = self.Phi = _identity_map()
        lam = [np.exp(2j * np.pi * tunes[0]), np.exp(-2j * np.pi * tunes[0]),
               np.exp(2j * np.pi * tunes[1]), np.exp(-2j * np.pi * tunes[1])]
        self.N = Map(*(Polynom([Term(lam[j], *(int(i == j) for i in range(4)))]) for j in range(4)))

    def norm_to_nf(self, *z):
        return self.Psi.substitute(*z)

    def nf_to_norm(self, *z):
        return self.Phi.substitute(*z)


def detuning_hamiltonian(c: float = 0.0) -> Polynom:
    """H = i (a J1 + b J2 + d J1^2 + e J1 J2) + i c (z1^2 z2* ... coupling)."""
    terms = [
        Term(1j * 1.9, 1, 1, 0, 0), Term(1j * 2.0, 0, 0, 1, 1),
        Term(1j * 0.05, 2, 2, 0, 0), Term(1j * 0.03, 1, 1, 1, 1),
    ]
    if c:
        terms += [Term(1j * c, 2, 0, 0, 1), Term(1j * c, 0, 2, 1, 0)]
    return Polynom(terms)


def tracked_orbits_from(z0: np.ndarray, turns: int, step) -> Orbits:
    """Orbits object filled by iterating ``step`` on normalised coordinates."""
    traj = [z0]
    for _ in range(turns - 1):
        traj.append(step(traj[-1]))
    traj = np.array(traj)  # (turns, 4, N)
    planes = _initial_planes(z0)
    n = z0.shape[1]
    norm = {f"{c}_norm": traj[:, i].T for i, c in enumerate(("x", "px", "y", "py"))}
    return Orbits(
        plane=planes, radius=np.hypot(z0[0], z0[1]) + np.hypot(z0[2], z0[3]),
        angle=np.zeros(n), lost=np.zeros(n, bool), num_valid_turns=np.full(n, turns),
        x=traj[:, 0].T, px=traj[:, 1].T, y=traj[:, 2].T, py=traj[:, 3].T, **norm,
    )


class TestHelpers:
    def test_distance_to_polylines(self, rng=np.random.default_rng(0)):
        lines = [rng.normal(size=(30, 2)), rng.normal(size=(5, 2)) + 3]
        pts = rng.normal(size=(50, 2)) * 2
        pts[3] = np.nan
        got = _distance_to_polylines(pts, lines)
        best = np.full(len(pts), np.inf)
        for ln in lines:
            for a, b in zip(ln[:-1], ln[1:]):
                d = b - a
                t = np.clip(((pts - a) @ d) / (d @ d), 0, 1)
                best = np.minimum(best, np.linalg.norm(pts - (a + t[:, None] * d), axis=1))
        assert np.isnan(got[3])
        ok = np.isfinite(pts[:, 0])
        np.testing.assert_allclose(got[ok], best[ok], atol=1e-12)

    def test_distance_on_vertices_is_zero(self):
        line = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        np.testing.assert_allclose(_distance_to_polylines(line, [line]), 0, atol=1e-15)
        np.testing.assert_allclose(
            _distance_to_polylines(np.array([[0.5, 0.3], [2.0, 0.0]]), [line]), [0.3, 1.0]
        )

    def test_initial_planes(self):
        # rounding-level amplitudes in the other plane are tolerated
        z = np.array([[1.0, 0, 3.0, 1e-17], [0.5, 0, 0, 0], [1e-15, 2.0, 0, 1], [0, 0, 0, 1]])
        np.testing.assert_array_equal(_initial_planes(z), ["x", "y", "x", "y"])

    @pytest.mark.parametrize(
        "z", [np.array([[1.0], [0], [0.1], [0]]), np.zeros((4, 1))]
    )
    def test_initial_planes_invalid(self, z):
        with pytest.raises(ValueError, match="one plane"):
            _initial_planes(z)


class TestHamiltonianOrbitsExact:
    """Identity transformations and an action-dependent H: circular orbits."""

    @pytest.fixture
    def setup(self):
        nf = StubNormalForm(detuning_hamiltonian())
        _, _, _, z0 = initial_conditions("both", sigma_limit=3, radial_step=1)
        return nf, z0

    def test_circles(self, setup):
        nf, z0 = setup
        ho = hamiltonian_orbits(nf, z0, grid_size=801)
        assert isinstance(ho, HamiltonianOrbits) and ho.num_particles == 36
        for i in range(ho.num_particles):
            assert len(ho.curves_nf[i]) == 1
            c = ho.curves_nf[i][0]
            k = 0 if ho.plane[i] == "x" else 2
            np.testing.assert_allclose(np.hypot(c[k], c[k + 1]), ho.radius[i], rtol=1e-4)
            # other plane held at its initial (zero) value
            assert np.all(c[2 - k] == 0) and np.all(c[3 - k] == 0)
            np.testing.assert_allclose(c[:, 0], c[:, -1])
            np.testing.assert_allclose(ho.curves_norm[i][0], c, atol=1e-15)

    def test_labels_from_array(self, setup):
        nf, z0 = setup
        ho = hamiltonian_orbits(nf, z0)
        planes, radius, angle, _ = initial_conditions("both", 3, 1)
        np.testing.assert_array_equal(ho.plane, planes)
        np.testing.assert_allclose(ho.radius, radius)
        np.testing.assert_allclose(ho.angle, angle, atol=1e-12)
        np.testing.assert_allclose(ho.z0_norm, z0)
        np.testing.assert_allclose(ho.z0_nf, z0, atol=1e-15)

    def test_h0(self, setup):
        nf, z0 = setup
        ho = hamiltonian_orbits(nf, z0)
        J1, J2 = (z0[0] ** 2 + z0[1] ** 2), (z0[2] ** 2 + z0[3] ** 2)
        np.testing.assert_allclose(ho.h0, 1.9 * J1 + 2.0 * J2 + 0.05 * J1**2 + 0.03 * J1 * J2)
        np.testing.assert_allclose(ho.hamiltonian(*z0), ho.h0)

    def test_select(self, setup):
        nf, z0 = setup
        ho = hamiltonian_orbits(nf, z0).select("y")
        assert ho.num_particles == 18 and np.all(ho.plane == "y")
        assert len(ho.curves_nf) == len(ho.curves_norm) == 18
        assert ho.z0_nf.shape == (4, 18)
        with pytest.raises(ValueError):
            ho.select("both")

    def test_accuracy_with_exact_flow(self, setup):
        # Track with the exact flow of H: rotation with action-dependent tunes
        nf, z0 = setup

        def step(z):
            x, px, y, py = z
            J1, J2 = x**2 + px**2, y**2 + py**2
            mu1, mu2 = 1.9 + 2 * 0.05 * J1 + 0.03 * J2, 2.0 + 0.03 * J1
            zeta1 = (x - 1j * px) * np.exp(1j * mu1)
            zeta2 = (y - 1j * py) * np.exp(1j * mu2)
            return np.real(np.array(complex_to_real(zeta1, np.conj(zeta1), zeta2, np.conj(zeta2))))

        orb = tracked_orbits_from(z0, 50, step)
        ho = hamiltonian_orbits(nf, orb, grid_size=801)
        assert np.nanmax(ho.energy_deviation(orb)) < 1e-12
        assert np.nanmax(ho.distance(orb)) < 1e-4
        assert np.nanmax(ho.distance(orb, "norm")) < 1e-4

    def test_orbits_must_match(self, setup):
        nf, z0 = setup
        ho = hamiltonian_orbits(nf, z0)
        other = tracked_orbits_from(z0 * 1.1, 3, lambda z: z)
        with pytest.raises(ValueError, match="initial conditions"):
            ho.energy_deviation(other)
        with pytest.raises(ValueError, match="initial conditions"):
            ho.distance(other)
        with pytest.raises(ValueError, match="coordinates"):
            ho.distance(tracked_orbits_from(z0, 3, lambda z: z), coordinates="phys")

    def test_coupling_warning(self, setup):
        nf = StubNormalForm(detuning_hamiltonian(c=0.2))
        z0 = np.array([[1.0], [0.0], [0.0], [0.0]])
        # with the other plane exactly at zero the coupling terms vanish
        hamiltonian_orbits(nf, z0)
        z0 = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        nf.Psi = Map(
            Polynom([Term(1.0, 1, 0, 0, 0), Term(0.1, 0, 0, 1, 0)]),
            Polynom([Term(1.0, 0, 1, 0, 0), Term(0.1, 0, 0, 0, 1)]),
            Polynom([Term(1.0, 0, 0, 1, 0)]),
            Polynom([Term(1.0, 0, 0, 0, 1)]),
        )
        with pytest.warns(UserWarning, match="couples the planes"):
            hamiltonian_orbits(nf, z0)

    def test_extent_and_open_curves(self, setup):
        nf, _ = setup
        z0 = np.array([[2.0], [0.0], [0.0], [0.0]])
        # the grid lies inside the circle of radius 2: no curve
        ho = hamiltonian_orbits(nf, z0, extent=1.0)
        assert ho.curves_nf[0] == []
        # the corners of the grid reach the circle: open arcs
        ho = hamiltonian_orbits(nf, z0, extent=1.6)
        assert len(ho.curves_nf[0]) == 4
        for c in ho.curves_nf[0]:
            np.testing.assert_allclose(np.hypot(c[0], c[1]), 2, rtol=1e-3)
            assert not np.allclose(c[:, 0], c[:, -1])
        ho = hamiltonian_orbits(nf, z0, extent=2.5)
        assert np.max(np.abs(ho.curves_nf[0][0][:2])) <= 2.5

    @pytest.mark.parametrize(
        "kwargs, match",
        [(dict(grid_size=2), "grid_size"), (dict(initial=np.zeros((3, 2))), "shape")],
    )
    def test_invalid(self, setup, kwargs, match):
        nf, z0 = setup
        args = dict(initial=z0)
        args.update(kwargs)
        with pytest.raises(ValueError, match=match):
            hamiltonian_orbits(nf, **args)

    def test_not_computed(self):
        from xnf4d.numerical_normal_form import NormalForm4D

        with pytest.raises(ValueError, match="not been computed"):
            hamiltonian_orbits(NormalForm4D(_identity_map(), 3), np.array([[1.0], [0], [0], [0]]))


class TestHamiltonianOrbitsLHC:
    def test_nominal_line(self, lhc_line, twiss, normal_form, orbits):
        ho = hamiltonian_orbits(normal_form, orbits)
        assert all(len(c) == 1 for c in ho.curves_nf)
        assert np.nanmax(ho.energy_deviation(orbits)) < 1e-4
        assert np.nanmax(ho.distance(orbits)) < 1e-3
        # horizontal particles: the vertical normal form amplitude is tiny
        sx = orbits.select("x")
        hx = ho.select("x")
        assert np.nanmax(hx.distance(sx, "norm")) < 1e-3
        # array input gives the same orbits
        z0 = np.array([orbits.x_norm[:, 0], orbits.px_norm[:, 0],
                       orbits.y_norm[:, 0], orbits.py_norm[:, 0]])
        ho2 = hamiltonian_orbits(normal_form, z0)
        np.testing.assert_array_equal(ho2.plane, ho.plane)
        np.testing.assert_allclose(ho2.radius, ho.radius, rtol=1e-12)
        np.testing.assert_allclose(ho2.h0, ho.h0)

    def test_fourth_order_islands(self, lhc_line):
        """Quasi-resonant normal form at 4 Qx = 1: the contours are island chains."""
        from xnf4d.one_turn_map import OneTurnMap4D

        line = lhc_line
        saved = {k: line.vars[k]._value for k in ("dqx.b1", "dqy.b1", "i_mo_b1")}
        try:
            line.vars["i_mo_b1"] = -26
            for _ in range(2):
                tw = line.twiss()
                line.vars["dqx.b1"] = line.vars["dqx.b1"]._value + 62.28 - tw.qx
                line.vars["dqy.b1"] = line.vars["dqy.b1"]._value + 60.31 - tw.qy
            otm = OneTurnMap4D(line, 8, nemitt_x=NEMITT, nemitt_y=NEMITT, seed=1)
            qy = otm.twiss.qy % 1
            nf = otm.calculate_normal_form(
                7, "quasiresonant", res_basis1=[4, 0],
                res_eig=[1j, -1j, np.exp(2j * np.pi * qy), np.exp(-2j * np.pi * qy)],
            )
            orb = track_orbits(line, NEMITT, NEMITT, plane="x", sigma_limit=7,
                               radial_step=1, num_turns=256, twiss=otm.twiss, angles=[0, 54])
        finally:
            for k, v in saved.items():
                line.vars[k] = v
        ho = hamiltonian_orbits(nf, orb)
        inside = orb.radius == 7      # in the islands
        inner = orb.radius <= 3       # well below them
        counts = np.array([len(c) for c in ho.curves_nf])
        # one curve per orbit, or one per island of the chain
        assert set(counts) <= {1, 4}, counts
        assert np.all(counts[inside] == 4) and np.all(counts[inner] == 1), counts
        dE = ho.energy_deviation(orb)
        assert np.nanmax(dE[inner]) < 1e-3
        assert np.nanmax(dE[inside]) < 0.3
        assert np.nanmax(ho.distance(orb)[inside]) < 0.6


class TestOrbitSelection:
    """Several orbits on the same level of h must not be mixed."""

    def test_two_circles_same_level(self):
        # h = 2 J - J^2 / 2 (J = x^2 + px^2) has a maximum at J = 2: the levels
        # below it consist of two circles, J = 1 and J = 3 have the same h
        nf = StubNormalForm(Polynom([Term(2j, 1, 1, 0, 0), Term(-0.5j, 2, 2, 0, 0)]))
        z0 = np.array([[1.0, np.sqrt(3.0)], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        ho = hamiltonian_orbits(nf, z0, grid_size=801, extent=2.5)
        assert ho.h0[0] == pytest.approx(ho.h0[1])
        for i, r in enumerate((1.0, np.sqrt(3.0))):
            assert len(ho.curves_nf[i]) == 1
            c = ho.curves_nf[i][0]
            np.testing.assert_allclose(np.hypot(c[0], c[1]), r, rtol=1e-3)

    def test_islands_core_and_outside(self):
        """
        Pendulum-like 4th-order resonance in the rotating frame:
        h = -J + J^2 / 2 + 0.05 Re(zeta^4), N a quarter-turn rotation.
        Four islands around J ~ 1.1 on the diagonals; a particle in the core
        and one outside the islands share the same h.
        """
        nf = StubNormalForm(
            Polynom([Term(-1j, 1, 1, 0, 0), Term(0.5j, 2, 2, 0, 0),
                     Term(0.025j, 4, 0, 0, 0), Term(0.025j, 0, 4, 0, 0)]),
            tunes=(0.25, 0.32),
        )
        J_core = 0.4
        h_core = -J_core + 0.55 * J_core**2           # on the x axis, Re(zeta^4) = J^2
        J_out = (1 + np.sqrt(1 + 2.2 * h_core)) / 1.1   # other root of 0.55 J^2 - J = h_core
        J_island = 0.8                               # on the diagonal
        r = np.sqrt([J_core, J_out, J_island])
        z0 = np.array([[r[0], r[1], r[2] / np.sqrt(2)], [0.0, 0.0, r[2] / np.sqrt(2)],
                       [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        ho = hamiltonian_orbits(nf, z0, grid_size=801, extent=1.6)
        assert ho.h0[0] == pytest.approx(ho.h0[1])
        # the whole level set has core, outer and nothing else at that level
        assert [len(c) for c in ho.curves_nf] == [1, 1, 4]
        for i in (0, 1):
            c = ho.curves_nf[i][0]
            # each keeps its own curve: the one through its initial point
            rr = np.hypot(c[0], c[1])
            assert np.min(np.abs(rr - r[i])) < 1e-3 and np.max(rr) < (1.1 if i == 0 else 2)
            assert (np.max(rr) < 1) == (i == 0)
        # the four island curves are rotated copies of each other
        centres = [np.angle(np.mean(c[0]) + 1j * np.mean(c[1]), deg=True) % 360
                   for c in ho.curves_nf[2]]
        np.testing.assert_allclose(sorted(centres), [45, 135, 225, 315], atol=1)
