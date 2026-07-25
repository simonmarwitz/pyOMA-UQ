"""
Tier 1 application tests — self-contained, always CI.

Tests that vars_definition() constructs correct PolyUQ-compatible variable
definitions for the archived full study.
"""
import numpy as np
import pytest
from polyuq import RandomVariable, MassFunction, PolyUQ


@pytest.fixture(scope="module")
def vars_all_stages():
    from pyoma_uq.studies.UQ_OMA import vars_definition
    return {
        1: vars_definition(stage=1),
        2: vars_definition(stage=2),
        3: vars_definition(stage=3),
    }


class TestVarsDefinition:
    def test_stage1_counts(self, vars_all_stages):
        vars_ale, vars_epi, arg_vars = vars_all_stages[1]
        assert len(vars_ale) == 2    # v_b, alpha
        assert len(vars_epi) == 2    # lamda, c

    def test_stage2_epi_count(self, vars_all_stages):
        vars_ale, vars_epi, arg_vars = vars_all_stages[2]
        # stage 2 adds n_locations, DTC, sensitivity_nominal, sensitivity_deviation,
        # spectral_noise_slope, sensor_noise_rms, range_estimation_duration,
        # range_estimation_margin, DAQ_noise_rms, decimation_factor,
        # anti_aliasing_cutoff_factor, quant_bit_factor, duration  => 13 + 2 stage1 = 15
        assert len(vars_epi) == 15

    def test_all_vars_have_correct_type(self, vars_all_stages):
        vars_ale, vars_epi, _ = vars_all_stages[3]
        for v in vars_ale + vars_epi:
            assert isinstance(v, (RandomVariable, MassFunction)), (
                f"{v.name} is neither RandomVariable nor MassFunction"
            )

    def test_primary_flags(self, vars_all_stages):
        vars_ale, vars_epi, arg_vars = vars_all_stages[2]
        # every name in arg_vars must correspond to a primary variable
        primary_names = {v.name for v in vars_ale + vars_epi if v.primary}
        for arg_name, var_name in arg_vars.items():
            assert var_name in primary_names, (
                f"arg_var '{arg_name}' maps to '{var_name}' which is not primary"
            )

    def test_v_b_weibull(self, vars_all_stages):
        vars_ale, _, _ = vars_all_stages[1]
        v_b = next(v for v in vars_ale if v.name == "v_b")
        assert isinstance(v_b, RandomVariable)

    def test_sample_qmc_stage2(self, vars_all_stages):
        vars_ale, vars_epi, _ = vars_all_stages[2]
        poly = PolyUQ(vars_ale, vars_epi, dim_ex="cartesian")
        poly.sample_qmc(N_mcs_ale=4, N_mcs_epi=9, check_discr=False)
        assert poly.inp_samp_prim is not None


class TestSensitiveVars:
    def test_sensitive_vars_subset(self, vars_all_stages):
        from pyoma_uq.studies.UQ_OMA import sensitive_vars
        _, vars_epi, _ = vars_all_stages[3]
        result = sensitive_vars("f_sc", vars_epi)
        epi_names = {v.name for v in vars_epi}
        for v in result:
            assert v.name in epi_names
