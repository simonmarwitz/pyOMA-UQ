'''Structural, wind and acquisition models of the model-based path.

Only :mod:`pyoma_uq.studies.UQ_OMA_weighted` (and the archived
:mod:`pyoma_uq.studies.UQ_OMA`) use these; the experimental path replaces all
of them with measured signals.

* :mod:`~pyoma_uq.models.mechanical` -- the guyed-mast structural model
  (with :mod:`~pyoma_uq.models.mechanical_fun`)
* :mod:`~pyoma_uq.models.turbulent_wind` -- the excitation
* :mod:`~pyoma_uq.models.acquisition` -- the sensing/acquisition chain
  (sensor response, measurement range, sampling, quantisation, DAQ noise)
* :mod:`~pyoma_uq.models.toy_response` -- a cheap synthetic response used by
  the development fixtures
'''
import logging
import sys

logging.basicConfig(stream=sys.stdout)
