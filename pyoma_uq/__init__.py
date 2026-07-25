'''
pyoma-uq -- polymorphic uncertainty quantification for operational modal
analysis, built on pyOMA (identification) and PolyUQ (uncertainty engine).

Two entry paths are supported, both driving the same weighted OMA estimator
through PolyUQ:

* **model-based, with synthetic measurements** -- a structural model
  (:mod:`pyoma_uq.models`) with numerical excitation and a sensing/acquisition
  chain produces the signals
  (:mod:`pyoma_uq.studies.UQ_OMA_weighted`);
* **experimental** -- measured signals enter directly, leaving only signal
  processing and system identification
  (:mod:`pyoma_uq.studies.UQ_OMA_experimental`).
'''
