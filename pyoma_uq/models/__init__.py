'''Data sources for the model-based path.

There is no FEM solver here. The model-based path consumes a response that the
user provides -- either a pre-computed modal/FRF archive
(:mod:`~pyoma_uq.models.modal_archive`) or any callable returning acceleration
histories -- and the ANSYS bridge that once generated those archives has been
removed, along with the wind-field generator and the archived full study.

* :mod:`~pyoma_uq.models.modal_archive` -- read a modal/FRF archive and
  synthesize transients from it (``ModalArchive.transient_ifrf``)
* :mod:`~pyoma_uq.models.toy_response` -- a compact synthetic response built
  from a stored modal solution, used by the development fixtures
* :mod:`~pyoma_uq.models.acquisition` -- the sensing/acquisition chain
  (sensor response, measurement range, sampling, quantisation, DAQ noise)
'''
import logging
import sys

logging.basicConfig(stream=sys.stdout)
