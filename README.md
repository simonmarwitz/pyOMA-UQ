# oma_uq

Application layer for [PolyUQ](https://github.com/simonmarwitz/PolyUQ) — a
structural health monitoring / operational modal analysis (OMA) case study
quantifying uncertainty in a guyed-mast wind turbine tower under wind and
ice loading.

`oma_uq` provides the structural models (`model/`) and case-study scripts
(`examples/`) that define the aleatory/epistemic variables and mapping
functions for the mast, and post-processes PolyUQ propagation results into
sensitivity, imprecision, and incompleteness estimates.

## Installation

```bash
pip install -e .
pip install -e ../PolyUQ  # or: pip install polyuq
```

Optional extras:

```bash
pip install -e .[hpc]   # ray/scikit-learn/simpleflock/psutil for distributed evaluation and clustering
pip install -e .[fem]   # requires a manual ANSYS PyMAPDL install; not on PyPI
pip install -e .[dev]   # test suite
```

## Contents

- `examples/UQ_OMA.py` — full three-stage OMA case study (wind field →
  structural response → virtual sensing/identification).
- `examples/UQ_Modal_Analytical.py` — analytical beam frequency/damping model.
- `examples/UQ_Modal_FEM.py` — ANSYS-based finite-element beam model.
- `examples/UQ_Acqui.py` — signal acquisition and quantization studies.
- `model/` — structural (`mechanical.py`), acquisition (`acquisition.py`),
  and wind-field (`turbulent_wind.py`) models, plus DSP/illustration
  notebooks.

Many of the notebooks and demo functions here were used to produce results
for the associated thesis/publications and reference pre-computed HPC
outputs; they cannot be fully re-run without access to the original cluster
data or the published [refodat dataset](https://doi.org/10.71758/refodat.46).

## License

GPL v3, see [LICENSE](LICENSE).
