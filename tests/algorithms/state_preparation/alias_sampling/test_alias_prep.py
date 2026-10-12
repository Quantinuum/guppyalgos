"""Alias Sampling Test."""

from selene_sim import Quest
from guppyalgos.testing import project_state_onto_bitstring

from math import ceil, log2
import numpy as np
import numpy.typing as npt
import pytest
from guppylang import guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.debug import state_output
from guppylang.std.quantum import discard, discard_array, qubit

from guppyalgos.primitives.subroutines.fanout import (
    fanout_basic,
    fanout_measurement_parity,
)
from guppyalgos.algorithms.state_preparation.alias_sampling import alias_samp_prep
from guppyalgos.algorithms.state_preparation.alias_sampling.alias_table import (
    discretize_distribution,
)
from guppyalgos.utils import qarray, int_to_bits


from typing import no_type_check


@pytest.mark.parametrize(
    ("n_prob_bits", "prob_dist", "precision", "fanout_op"),
    [
        (2, np.array([0.0, 0.5, 0.25, 0.25]), 0.25, fanout_basic),
        (2, np.array([0.25, 0.5, 0.25, 0.0]), 0.25, fanout_basic),
        (2, np.array([0.0, 0.5, 0.5, 0]), 0.25, fanout_basic),
        (2, np.array([0.5, 0.25, 0.25]), 0.25, fanout_basic),
        (
            3,
            np.array([0.0, 0.5, 0.25, 0.25, 0.0, 0.5, 0.25, 0.25]) / 2,
            0.125,
            fanout_basic,
        ),
        (2, np.array([0.0, 0.5, 0.25, 0.25]), 0.25, fanout_measurement_parity),
    ],
)
def test_alias_sampling(
    n_prob_bits: int,
    prob_dist: npt.NDArray[np.float64],
    precision: float,
    fanout_op: GuppyFunctionDefinition,
) -> None:
    """Test alias sampling circuit gives correct amplitudes."""
    n_qubits = ceil(log2(len(prob_dist)))
    alias_samp_prep_box = alias_samp_prep(
        prob_dist,
        precision,
        fanout_op=fanout_op,
    )
    n_fixed_point_bits = ceil(log2(1 / precision))
    disc_dist = discretize_distribution(prob_dist, n_fixed_point_bits)

    @guppy
    @no_type_check
    def main() -> None:
        index_reg = qarray(n_qubits)
        alternative_val_reg = qarray(n_qubits)
        compare_alt_reg = qarray(n_prob_bits)
        keep_prob_reg = qarray(n_prob_bits)
        compare_out = qubit()
        alias_samp_prep_box(
            index_reg,
            alternative_val_reg,
            keep_prob_reg,
            compare_alt_reg,
            compare_out,
            False,
        )
        state_output("result_state", index_reg)
        discard_array(alternative_val_reg)
        discard_array(keep_prob_reg)
        discard_array(compare_alt_reg)
        discard_array(index_reg)
        discard(compare_out)

    total_qubits = 3 * n_qubits + 2 * n_prob_bits + 5 + int(np.ceil((n_qubits - 3) / 2))
    res = main.emulator(total_qubits).run()

    states = Quest.extract_states_dict(res.results[0].entries)
    for i in range(len(disc_dist)):
        projected_onto_index = project_state_onto_bitstring(
            states["result_state"], int_to_bits(i, n_qubits)
        )
        np.testing.assert_allclose(
            projected_onto_index.probability, disc_dist[i], atol=1e-14
        )


@pytest.mark.parametrize(
    ("n_prob_bits", "prob_dist", "precision"),
    [
        (2, np.array([0.0, 0.5, 0.25, 0.25]), 0.25),
        (2, np.array([0.25, 0.5, 0.25, 0.0]), 0.25),
        (2, np.array([0.0, 0.5, 0.5, 0]), 0.25),
        (2, np.array([0.5, 0.25, 0.25]), 0.25),
        (3, np.array([0.0, 0.5, 0.25, 0.25, 0.0, 0.5, 0.25, 0.25]) / 2, 0.125),
    ],
)
def test_alias_sampling_inverse(
    n_prob_bits: int, prob_dist: npt.NDArray[np.float64], precision: float
) -> None:
    """Test alias sampling circuit inverts correctly."""
    n_qubits = ceil(log2(len(prob_dist)))
    alias_samp_prep_box = alias_samp_prep(prob_dist, precision)

    @guppy
    @no_type_check
    def main() -> None:
        index_reg = qarray(n_qubits)
        alternative_val_reg = qarray(n_qubits)
        compare_alt_reg = qarray(n_prob_bits)
        keep_prob_reg = qarray(n_prob_bits)
        compare_out = qubit()
        alias_samp_prep_box(
            index_reg,
            alternative_val_reg,
            keep_prob_reg,
            compare_alt_reg,
            compare_out,
            False,
        )
        alias_samp_prep_box(
            index_reg,
            alternative_val_reg,
            keep_prob_reg,
            compare_alt_reg,
            compare_out,
            True,
        )
        state_output("index_reg", index_reg)
        state_output("alternative_val_reg", alternative_val_reg)
        state_output("keep_prob_reg", keep_prob_reg)
        state_output("compare_out", compare_out)
        discard_array(alternative_val_reg)
        discard_array(keep_prob_reg)
        discard_array(compare_alt_reg)
        discard_array(index_reg)
        discard(compare_out)

    total_qubits = 3 * n_qubits + 2 * n_prob_bits + 5 + int(np.ceil((n_qubits - 3) / 2))
    res = main.emulator(total_qubits).run()

    for state in res.partial_state_dicts()[0].values():
        assert state.as_single_state()[0] == 1


def test_alias_workspace_is_not_clean_after_select() -> None:
    """UNPREPARE after SELECT can leave alias workspace entangled."""
    import zixy.qubit.pauli as zqp

    from guppyalgos.algorithms.block_encoding.lcu import (
        LCUData,
        build_unary_iteration_select,
    )

    data = LCUData.from_hamiltonian(
        zqp.RealTermSum.from_str(
            "(0.25, Z0), (-0.125, X1), (0.125, Y0 Y1), (0.125, Z0 X1), "
            "(0.125, X0), (0.125, Z1), (0.0625, X0 Z1), (-0.0625, Z0 Z1)",
            2,
        )
    )
    prepare = alias_samp_prep(np.abs(data.coeffs) / data.l1_norm, precision=1 / 16)
    select = build_unary_iteration_select(data)

    @guppy
    @no_type_check
    def main() -> None:
        index = qarray(3)
        alternative = qarray(3)
        keep = qarray(4)
        comparison = qarray(4)
        flag = qubit()
        qreg = qarray(2)
        prepare(index, alternative, keep, comparison, flag, False)
        select(index, qreg)
        prepare(index, alternative, keep, comparison, flag, True)
        state_output("comparison", comparison)
        discard_array(index)
        discard_array(alternative)
        discard_array(keep)
        discard_array(comparison)
        discard(flag)
        discard_array(qreg)

    result = main.emulator(22).run()
    state = Quest.extract_states_dict(result.results[0].entries)["comparison"]
    probability_zero = project_state_onto_bitstring(state, [False] * 4).probability
    np.testing.assert_allclose(probability_zero, 13 / 16, atol=1e-10)
