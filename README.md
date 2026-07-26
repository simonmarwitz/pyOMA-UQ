# pyoma-uq

Polymorphic uncertainty quantification for operational modal analysis — an
extension module to [pyOMA](https://github.com/simonmarwitz/pyOMA), driven by
the [PolyUQ](https://github.com/simonmarwitz/PolyUQ) uncertainty engine.

OMA identifies modal parameters from ambient vibration data, but the answer
depends on choices nobody can pin down exactly: the analysis band, the number
of block rows, the model order, and the excitation the structure happened to
see while it was measured. `pyoma-uq` propagates those *polymorphic*
uncertainties — aleatory variability, imprecision and incompleteness — through
the identification itself rather than around it: PolyUQ's
Incompleteness-conditioned importance weights become the block weights of a
weighted subspace identification, so the aleatory dimension is folded into the
estimator instead of being condensed after `N_ale` separate runs.

## The two supported paths

| | model-based | experimental |
|---|---|---|
| module | `pyoma_uq.studies.UQ_OMA_weighted` | `pyoma_uq.studies.UQ_OMA_experimental` |
| signals | synthesised from a **response you provide**, through the acquisition chain (`pyoma_uq.models.acquisition`) | measured |
| stages | response provider → acquisition → signal processing → identification | signal processing → identification |
| aleatory realisations | sampled excitation levels | the measured data blocks, weighted by their observed response level |
| mode assignment | global pole clustering (OPTICS) | pairing against a known baseline mode set, inside the estimator |

Both end in the same statistic-level step: per mode, the epistemic samples are
fitted with a surrogate and interval-optimised over the combined Imprecision ×
Incompleteness hypercubes (`PolyUQ.to_statistic_level` → `estimate_imp`).

**There is no FEM solver in this package.** The model-based path takes a
response *provider* — the structural model is yours, and only its output
crosses into `pyoma-uq`. (Earlier versions embedded an ANSYS bridge and a
wind-field generator; both were removed, along with the archived
per-realisation study that depended on them.)

---

## Path 1 — model-based, from a response you provide

A provider is any callable `(id_ale, v_b) -> (t_vals, accel, nodes)` with
`accel` shaped `(n_timesteps, n_nodes, 2)`. Two are supplied:

```python
from pyoma_uq.studies.UQ_OMA_weighted import (
    ToyResponseProvider,      # synthesises from a stored modal solution
    FRFResponseProvider,      # convolves a stored compliance FRF
    run_weighted_uq_pipeline,
)

# reads a modal/FRF archive (pyoma_uq.models.modal_archive.ModalArchive)
poly_uq, pole_db, theta, results = run_weighted_uq_pipeline(
    result_dir='runs/model_based',
    mech_npz='samples/mechanical.npz',
    N_ale=100, N_epi=200,
    method='sc',              # 'sc' cov-SSI | 'sd' data-SSI | 'cf' pLSCF
    weighting='build',        # 'build' moves the point estimate | 'posthoc'
    provider_cls=FRFResponseProvider,
)
```

Bringing your own structure means writing one class:

```python
import numpy as np

class MyResponseProvider:
    """Whatever produces acceleration histories: an FE solver, a rig, a file."""

    def __init__(self, result_dir, mech_npz):   # signature fixed by run_lattice
        self.result_dir, self.mech_npz = result_dir, mech_npz

    def __call__(self, id_ale, v_b):
        # id_ale identifies the aleatory realisation -- derive seeds from it so
        # a rerun reproduces; v_b is that realisation's excitation level.
        t_vals = np.arange(0, 3600, 1 / 70)
        accel = my_solver(level=v_b, seed=hash(id_ale) % 2**32)  # (N, n_nodes, 2)
        nodes = np.arange(1, accel.shape[1] + 1)
        return t_vals, accel, nodes

run_weighted_uq_pipeline(..., provider_cls=MyResponseProvider)
```

Cache responses per `id_ale` yourself if they are expensive — both bundled
providers write `<result_dir>/<id_ale>/response.npz` and reuse it.

## Path 2 — experimental, from measured signals

Nothing is synthesised: the measured record is split into blocks, and each
block's observed response level is the aleatory realisation.

```python
from pyoma_uq.studies.UQ_OMA_experimental import (
    reproduce_baseline, feasibility_report, run_experimental_pipeline,
)

# 1. sanity-check the harness against a known result first
table = reproduce_baseline(band='low')        # pairs against archived modes
print(table.sort_values('n_paired').tail(1))

# 2. cost nothing to see how many epistemic samples survive the screen
per_sample, per_hypercube = feasibility_report(poly_uq)
print(f'{per_sample.ok.mean():.0%} feasible, '
      f'{(per_hypercube.n_feasible == 0).sum()} empty hypercubes')

# 3. the study
poly_uq, baseline, coverage, results = run_experimental_pipeline(
    'runs/experimental', weighting='build', N_epi=1500)
```

Adapting it to another structure means supplying three things, all in
`UQ_OMA_experimental`: a `load_asc`-style reader bound via `install_loader()`,
the setup/channel config paths in `SETUPS`, and a baseline mode set for
`load_baseline_modes()` to pair against. The variable definitions in
`vars_definition_experimental()` carry a comment per focal set explaining what
grounds it — read those before changing the ranges.

### Two constraints worth knowing before you change variables

* **Imprecision focals must tile or nest over the variable's support.**
  `sample_qmc` draws uniformly over the support hull, so a gap between two
  focal sets is dead sampling volume and leaves hypercubes empty.
* **Don't sample quantities that are derived.** The decimation factor follows
  from the band; sampling it independently made the anti-alias constraint
  reject 90 % of samples. Sample the oversampling ratio and derive the rest
  (`resolve_decimation`).

## Installation

```bash
pip install -e .
pip install -e ../PolyUQ   # or: pip install polyuq
```

Optional extras:

```bash
pip install -e .[hpc]   # ray / scikit-learn / simpleflock / psutil
pip install -e .[dev]   # test suite
```

## Command line

The measurement data location defaults to `/home/womo1998/Projects/2019_Schwabach`
and can be overridden with `$SCHWABACH_DIR`.

```bash
# reproduce the published 2019 identification and pair it against the
# archived, manually selected modes
python -m pyoma_uq.studies.UQ_OMA_experimental --reproduce-baseline --band low

# dry-run the acceptance-rejection screen (milliseconds) to size N_epi
python -m pyoma_uq.studies.UQ_OMA_experimental --feasibility-only --n-epi 1000

# the full study
python -m pyoma_uq.studies.UQ_OMA_experimental <result_dir> \
    --weighting build --n-epi 1000
```

The identification cost grows roughly cubically with the number of block rows,
so a full sweep is a cluster job; `--feasibility-only` reports the sampled
`num_block_rows` distribution the budget follows from.

## Tests

```bash
pytest -m "not slow"    # fast: pipeline logic against stand-in estimators
pytest                  # adds the end-to-end identifications
```

Markers: `slow` (minutes), `data` (needs measurement data), `hpc` (needs a
cluster environment).

## License

GPL v3, see [LICENSE](LICENSE).
