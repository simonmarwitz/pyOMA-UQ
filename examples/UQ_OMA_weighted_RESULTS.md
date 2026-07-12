# Weighted-OMA pipeline — development-run results and dissertation notes

Implementation of the outlook "Toward OMA-Specific PolyUQ Processing Strategies"
(`chapter_main.tex`, Algorithm `alg:proposed`): OMA acts as the weighted statistical
estimator Θ instead of running once per `(aleatory, epistemic)` sample pair.
Code: `examples/UQ_OMA_weighted.py` (this repo, branch `feature/oma-weighted-estimator`),
pyOMA `feature/weighted-subspace` (weighted/externally-fed subspace estimation in
`VarSSIRef`), PolyUQ `feature/from-propagated-samples`
(`PolyUQ.from_propagated_samples`). Run artifacts:
`/home/womo1998/Projects/uq_oma_a/weighted_dev/`; validation:
`examples/validate_weighted_run.py`.

## Study definition (reduced, local, development data)

- Development data synthesized from the mast's real modal solution
  (`samples/mechanical.npz`, 38 non-classical modes) via a compact
  modal-superposition FRF (`model/toy_response.py`) — same physical modes as the
  original case study, so the known-mode anchor of Table `tab:modal_params` holds.
- N_ale = 100 × 300 s records (independent `v_b` draws), N_epi = 200.
- Variables: Imprecision `n_locations`, `model_order`, `tau_max`,
  `decimation_factor` (18 Imprecision hypercubes); Incompleteness secondary
  `c_vb`, `lamda_vb`, primary `DAQ_noise_rms` (4 Incompleteness hypercubes)
  → 72 combined focal intervals per statistic. All other acquisition parameters
  fixed at their highest-mass focal representatives.
- Cost: 20 000 cheap lattice cells (response → acquisition → correlation
  functions) + **200 weighted VarSSIRef identifications** total, all local —
  versus 10⁶ full OMA runs per method in the original per-sample study.

## Headline validation results (run of 2026-07-11/12, seed 1509)

1. **Degeneracy confirmed empirically.** No Imprecision variable has focal bounds
   formed by secondary-Variability samples, so the Incompleteness-conditioned
   weights are identical across all 18 Imprecision hypercubes: the per-hypercube
   loop deduplicated to exactly **1 distinct identification per epistemic
   sample** (200 total), as predicted by the algorithm analysis.
2. **All 13 clusters match known mast modes**, median |Δf/f| ≤ 0.05 % (worst
   0.45 % for the sparsely-found 0.163 Hz mode): 0.157, 0.163, 0.315, 0.335,
   0.580, 0.604, 1.200, 1.249, 2.011, 2.097, 3.035, 3.167, 4.285 Hz. No spurious
   clusters. The 0.179/0.180 Hz TMD pair (ζ ≈ 9 %) is absent, as expected — it is
   not OMA-resolvable from 300-s ambient records. Found-fractions 10–99 % of
   epistemic samples; low fractions are physically explained (close-pair
   resolution at 300 s; decimation-dependent Nyquist for the 3–4.3 Hz modes).
3. **Propagated uncertainties sane.** Kish n_eff = 41–47 (of 100) across all
   epistemic samples; 100 % of per-pole std_f and std_d finite and non-negative,
   magnitudes 0.03–0.30 of the respective frequency spacing.
4. **Statistic-level focal intervals** (72 per mode/quantity, RBF surrogate +
   genetic interval optimization via `PolyUQ.estimate_imp` on
   `from_propagated_samples` instances): every focal ordered (lo ≤ hi),
   high-mass focals nested inside the overall envelope, masses sum to 1.
5. **Qualitative agreement with the archived per-sample study**
   (`estimations/f_sc-*/polyuq_avg_inc.npz`): the weighted frequency envelopes
   overlap the archived ones for every common mode and are consistently
   *narrower* (e.g. mode at 1.1995 Hz: weighted [1.1993, 1.2001] vs archived
   [1.1969, 1.2022]) — plausible since the weighted estimator averages 100
   records before identification instead of propagating single-record scatter.
   Numeric agreement is not expected (different data realization, estimator,
   and variable set); same modes + overlapping magnitudes + consistent
   hypercube-effect ordering is the claimed result.
6. **Damping medians** track the known values well above 0.5 Hz (e.g. 1.1995 Hz:
   0.109 % vs 0.107 %; 0.6038 Hz: 0.183 % vs 0.166 %) and degrade for the first
   bending pair (2.0–2.7 % vs 1.4–2.0 %) — the familiar low-frequency damping
   bias of short ambient records, not a method artifact.
7. **Weight-sensitivity monotonicity**: freezing `c_vb` = 2.28 and sweeping
   `lamda_vb` 5.618 → 6.0 shifts the weighted-mean `v_b` strictly monotonically
   (6.50 → 6.88), confirming the Incompleteness conditioning acts on the weights
   as designed.

## Clarifications / deviations to record in the dissertation

- **Erratum for Algorithm line ~2154**: the surrogate input set is written
  `x_t ⊆ {X^i,p, X^c,q}`; the secondary-Incompleteness values X^c,s
  (`c_vb`, `lamda_vb`) must also be included, otherwise the interval
  optimization within H^c_r is not well-defined. The implementation uses
  x_t = [4 Imprecision vars] + [c_vb, lamda_vb, DAQ_noise_rms].
- **Sensor setups are seeded from the epistemic id** (the original
  `stage2mapping` used the aleatory id). Required by the algorithm's premise
  that spectral-estimator configuration is constant along the aleatory axis at
  fixed n_e, so the N_ale correlation matrices per epistemic sample are
  shape-consistent and averageable.
- **Weighted covariance**: weights renormalized to Σw = 1, Kish
  n_eff = 1/Σw²; T-matrix columns √(wₙ)(xₙ − x̄_w)/√(n_eff(n_eff − 1)) — reduces
  bit-for-bit to the uniform formula at wₙ = 1/N. Only
  `subspace_method='covariance'` + `variance_algo='fast'`; the slow-path
  weighted formula S_w/(n_eff − 1) is documented but guarded with
  `NotImplementedError`.
- **Global clustering** across all (n_e, q) cells (identical to per-hypercube
  clustering here due to the degeneracy).

## Future work (not implemented)

**First-order weight perturbation.** The fast variance algorithm's Jacobian
chain maps subspace-matrix perturbations to modal-parameter perturbations, and
under renormalization d(x̄_w)/d(wₙ) = xₙ − x̄_w — so dΘ/dwₙ is obtainable from
already-computed quantities. This would enable gradient-based Incompleteness
optimization directly on the weights (bypassing the statistic-level surrogate)
and analytic propagation of weight uncertainty. Derivation and validation are a
research task of their own.

**Mode-shape statistics.** Per-pole `std_mode_shapes` are carried through the
pipeline but no surrogate/interval optimization is run on shapes
(vector-valued; separate design problem).

**Real-data reproduction.** `stage2corr_mapping` accepts a response-provider
callable; swapping the development generator for the archived cluster
responses/FRF re-enables the original data path without interface changes.
