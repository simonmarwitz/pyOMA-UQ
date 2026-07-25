'''Case studies: the two supported entry paths of pyoma-uq.

``UQ_OMA_weighted``
    Model-based, with synthetic measurements: a structural model with
    numerical excitation and a sensing/acquisition model feeds the weighted
    OMA estimator.
``UQ_OMA_experimental``
    Experimental: measured signals feed the same estimator through signal
    processing and system identification only.
``UQ_OMA``
    The full per-realisation reference study. Reads archived cluster outputs
    and is *not* reproducible without the published dataset
    (https://doi.org/10.71758/refodat.46); kept because it is the reference
    the two supported paths are compared against.
'''
