# xnf4d

4D normal forms (Bazzani et al., https://cds.cern.ch/record/262179) of one-turn
maps for non-linear beam dynamics, compatible with xsuite. Main use case:
non-linearly matched beam distributions for collimation studies (e.g. HL-LHC with
beam-beam, primary collimators at 6.7 or 8.5 sigma; LHC with phase knob changes 
resulting in stable islands at large amplitudes; etc.).

## Rules from the user (always follow)

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

- **Think before coding!** Don't assume. Don't hide confusion. Surface tradeoffs. 
  Before implementing:
  - State your assumptions explicitly. If uncertain, ask.
  - If multiple interpretations exist, present them - don't pick silently.
  - If a simpler approach exists, say so. Push back when warranted.
  - If something is unclear, stop. Name what's confusing. Ask.
- **Simplicity first!** Minimum code that solves the problem. Nothing speculative.
  - No features beyond what was asked.
  - No abstractions for single-use code.
  - No "flexibility" or "configurability" that wasn't requested.
  - No error handling for impossible scenarios.
  - If you write 200 lines and it could be 50, rewrite it.
  Ask yourself: "Would a senior scientist/engineer say this is overcomplicated?" If yes, simplify.
- **Surgical Changes.** Touch only what you must. Clean up only your own mess. 
  When editing existing code:
  - Don't "improve" adjacent code, comments, or formatting, unless explicitly asked.
  - Don't refactor things that aren't broken.
  - Match existing style, even if you'd do it differently.
  - If you notice unrelated dead code, mention it - don't delete it, unless 
    explicitly asked.
  When your changes create orphans:
  - Remove imports/variables/functions that YOUR changes made unused.
  - Don't remove pre-existing dead code unless asked.
  The test: Every changed line should trace directly to the user's request.
- **Goal-Driven Execution.** Define success criteria. Transform tasks into 
  verifiable goals:
  - "Add validation" -> "Write tests for invalid inputs, then make them pass"
  - "Fix the bug" -> "Write a test that reproduces it, then make it pass"
  - "Refactor X" -> "Ensure tests pass before and after"
  For multi-step tasks, state a brief plan:
  ```
  1. [Step] -> verify: [check]
  2. [Step] -> verify: [check]
  3. [Step] -> verify: [check]
  ```
  Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.
- **Do not modify the mathematics** of the normal form. Psi, Phi, U must stay
  symplectic (to order); the one-turn map must stay correct and symplectic.
  Speed-ups are welcome.
- Treat the xtrack line as a **black box that supports tracking** (no TPSA, no
  inspection of its elements); linear information only from `line.twiss()`.
- No dependence on `xnlbd`. Never modify `xtrack`, `xpart`, `xcoll` or any other 
  xsuite package; work only in `xnf4d`.
- Conda env `xsuite_env`: use it to run/test, **never install or modify anything
  in it** (xnf4d is installed there in editable mode by the user).
- Keep the implementation clean; plotting is left to the user (examples and
  notebooks only).
- Unequal emittances (`nemitt_x != nemitt_y`) are parked: the per-plane scaling
  of normalised coordinates is not canonical when the map couples the planes.
  Don't work on it unless asked.

## Commands

```bash
conda run -n xsuite_env python -m pytest -q          # ~530 tests, ~3.5 min
python examples/pencil_on_collimator.py              # from the repo root
```

`conda run` does not forward stdin: write scripts to a file (scratchpad), don't
pipe them. Long HL-LHC runs (line ~45k elements) belong in the background.

## Layout

- `xnf4d/polynom.py`: `Term`, `Polynom`, `Map` (4 complex variables).
- `xnf4d/numerical_normal_form.py`: `NormalForm4D` (`res_case` =
  `"nonresonant" | "resonant" | "quasiresonant"`, `res_eig`, `res_basis1/2`;
  `compute_normal_form`, `norm_to_nf` (Psi), `nf_to_norm` (Phi), `H`, `N`,
  `to_json`/`from_json`). Complex coordinates `z1 = x - i px`.
- `xnf4d/one_turn_map.py`: `OneTurnMap4D` – fits a Lie map
  `R o exp(:f3:) o ... o exp(:f_{n+1}:)` to one-turn tracking.
- `xnf4d/orbits.py`: `track_orbits`, `initial_conditions`, `Orbits`,
  `hamiltonian_orbits` (2D contours, other plane frozen), `hamiltonian_tori`
  (4D invariant tori), `HamiltonianTori`.
- `xnf4d/distributions.py`: matched distributions and seams.
- `examples/`: pencil, halo, Gaussian bunch, q-Gaussian bunches at the LHC TCP
  (`test_data/levelling.26_b1.json.gz`, element `tcp.c6l7.b1`).
- Notebooks (user's, not tracked in git): `phase_knob_example.ipynb` (LHC 4th-order
  islands), `hl_bb.ipynb` (HL-LHC with beam-beam, `tcp.d6l7.b1`, 3 stitched maps).
- `pyproject.toml`: extras `xsuite`, `examples`, `test`; version 0.1.0.

## Key conventions and implementation notes

### Normalised coordinates
`xtrack` convention: `W^{-1} (x - x_co)` divided by `sqrt(nemitt / (beta0 gamma0))`.
W comes from the coupled eigenvectors, so **linear coupling is fully absorbed in
W**: the linear map in these coordinates is exactly two rotations (normal modes).
x_norm/y_norm are mode coordinates, not physical x/y.

### OneTurnMap4D
- Linear part fixed from twiss (`R = W^-1 M W`, forced to an exact rotation);
  only the non-linear generators (degrees 3..n+1) are fitted (least squares,
  QR-reduced, exact Jacobian).
- Sample uniform in Jx, Jy up to `sigma_limit**2/2` (per plane); `sigma_min`
  removes the core `2(Jx+Jy) < sigma_min**2` (shell fits). With `sigma_min=0` the
  random stream is unchanged from before.
- `midplane_symmetry="auto"` (default): tracks mirrored particles S z
  (S = diag(1,1,-1,-1)), measures the asymmetry, fits the symmetric model
  (generators even in (y, py)) and keeps it only if the RMS asymmetry is no
  larger than that fit's error on the symmetric data; else refits unconstrained.
  `z0`/`z1` then contain the mirrored particles too. `True`/`False` force it.
- `calculate_normal_form` warns if `max_map_order < max_nf_order + 1`.
- The fit is global: enlarging the region degrades the core. Beam-beam
  (`(1 - exp(-r^2/2))/r`) is far from polynomial: one map cannot be accurate
  from 1 to 7 sigma, raising the order helps little; use several fit ranges.

### Normal form choice (lessons)
- Non-resonant NF diverges near resonances (small denominators); higher NF order
  is worse at large amplitude. Phi is Psi's inverse only to order: invert Psi by
  Newton when exactness matters (`distributions._Transforms.psi_inverse`).
- HL-LHC with beam-beam: tunes with bb ~0.308/0.312, tracked tunes locked at
  Qx = Qy at all amplitudes -> use **quasi-resonant (1,-1)** with
  `res_eig` at `q_res = (qx%1 + qy%1)/2`. Invariants are h and K = A1 + A2.
  Beyond ~4 sigma in y the 3rd-order resonance (tune -> 1/3 with y amplitude)
  limits any single-resonance NF; double resonances have no tori and no usable
  invariant for distributions.
- With one resonance (n, m): invariants h and `m A1 - n A2`.

### hamiltonian_tori
Reduces the single-resonance motion at fixed K to 1 DOF: contour h = h0 in the
complex plane of the oscillator with the smaller action (other action from K,
phase fixed), Newton refinement onto the level, sweep of the free phase
(m, -n), map back with Newton inverse of Psi. Stepwise zoom (factor 4) for
reduced orbits too small for the grid; points equally spaced in both torus
directions (`num_points` total). Accepts a list of NFs + `seams` (band chosen by
each NF's rho, ties broken by |z0|); `tori.region`, `select("x"/"y")`,
`points(i)`, `distance(orbits)` (4D, KD-tree).

### distributions
- `generate_pencil_on_collimator` (xcoll-like interface; user passes NF at the
  collimator, plane, jaw in metres): accept if d(z) >= 0 and torus depth
  (max over both phases, exact tori via Newton) <= spread. Non-resonant NF only.
- `generate_2D_uniform_circular_sector`, `generate_matched_gaussian_bunch`,
  `generate_matched_qgaussian_bunch(q, beta, fact, sigma_cut_x/y, skew_cut, ...)`:
  sample in NF coordinates, map with Psi^-1. Exactly matched iff the density is
  a function of the NF invariants: Gaussian and non-factorisable q-Gaussian
  (function of A1 + A2) are matched with non-resonant and (1,-1) NFs; other
  cases warn ("approximately matched"); double resonances rejected.
  q-Gaussians = Student t (exact sampling, 1 <= q < 3); emittance is only the
  scale of normalised coordinates; cuts are in normalised coordinates.
- **Stitching**: bunch functions accept `normal_form=[nf0, nf1, ...]` (2-4) with
  `seams=[rho1, ...]` on rho = sqrt(A1 + A2); only NFs keeping A1 + A2 invariant,
  only densities of A1 + A2. `seam_mismatch(inner, outer, radii)` measures the
  seam layer (`summary()`, `best_radius`).

### Measured on HL-LHC (`hl_bb.ipynb` settings)
- Fits 0-4.5 / 3-5.5 / 4-8.5 sigma, seams 3.5 / 4.5: tori much better than a single
  fit below ~4.5 sigma (y at 3 sigma: 0.04 vs 0.29 sigma), similar above 5 sigma.
  NF order 3 better in y, order 7 better in x.
- Seams core|mid at 3.5 sigma: rms 0.008 sigma; mid|outer at 5 sigma: 0.06 sigma.
- Tail shell (4.5-6.5 sigma) tracked 200 turns: linear matching relaxes by 14x
  the sampling noise; non-linearly matched stays within noise.
- Do not match with beam-beam off and switch it on abruptly (beta/tune mismatch,
  breathing and filamentation during the first turns).

## Open items
- Pencil with resonant NFs (needs torus depth on resonant tori).
- Unequal emittances (see rules).
- Slowest test: `TestHamiltonianTori::test_small_orbit_far_from_start_amplitude` (~40 s).
