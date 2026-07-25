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
| signals | synthesised: structural model + numerical excitation + sensing/acquisition chain (`pyoma_uq.models`) | measured |
| stages | model → excitation → acquisition → signal processing → identification | signal processing → identification |
| aleatory realisations | sampled wind speeds | the measured data blocks, weighted by their observed response level |
| mode assignment | global pole clustering (OPTICS) | pairing against a known baseline mode set, inside the estimator |

Both end in the same statistic-level step: per mode, the epistemic samples are
fitted with a surrogate and interval-optimised over the combined Imprecision ×
Incompleteness hypercubes (`PolyUQ.to_statistic_level` → `estimate_imp`).

`pyoma_uq.studies.UQ_OMA` is the full per-realisation reference study the two
paths are compared against. It reads archived cluster outputs and is **not**
reproducible without the published
[dataset](https://doi.org/10.71758/refodat.46).

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

## Running the experimental study

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
