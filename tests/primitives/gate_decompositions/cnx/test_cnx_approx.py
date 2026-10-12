"""Test approximate CNX gate."""

from math import ceil, log2
from typing import no_type_check

import numpy as np
import pytest
from guppylang.decorator import guppy
from guppylang.std.debug import state_output
from guppylang.std.qsystem.random import RNG
from guppylang.std.quantum import discard, discard_array, h, qubit, x
from selene_sim import Quest

from guppyalgos.primitives.gate_decompositions.cnx.cnx_approx import cnx_approx
from guppyalgos.utils import int_to_bits, qarray
from guppyalgos.testing import (
    get_total_state_on_only_specified_registers,
    project_state_onto_bitstring,
)


@pytest.mark.parametrize("n_ctrl", [2, 3, 4, 5, 6, 7])
def test_cnx_approx_non_deterministic(n_ctrl: int) -> None:
    """Test cnx_approx with an average probability check."""
    epsilon = 0.1
    cnx_approx_fn = cnx_approx(n_ctrl, epsilon)

    k = min(ceil(log2(1 / epsilon)) + 2, n_ctrl)
    n_ancillas = (k - 2) // 2 + (k - 2) % 2 if k > 2 else 0

    @guppy
    @no_type_check
    def main() -> None:
        target = qubit()
        control_register = qarray(n_ctrl)

        for c in range(n_ctrl):
            h(control_register[c])

        rng = RNG(1)
        cnx_approx_fn(control_register, target, rng)

        state_output("target", target)
        state_output("controls", control_register)

        rng.discard()
        discard_array(control_register)
        discard(target)

    res = main.emulator(n_ctrl + 1 + n_ancillas).run()
    states = Quest.extract_states_dict(res.results[0].entries)
    non_work_state, spec_qubits_dict = get_total_state_on_only_specified_registers(
        states, ["target", "controls"]
    )
    control_qubits = spec_qubits_dict["controls"]
    target_qubit = spec_qubits_dict["target"]
    control_bitstrings = [int_to_bits(i, n_ctrl) for i in range(2**n_ctrl)]
    target_zero_probs = []

    for bitstring in control_bitstrings:
        non_work_state.specified_qubits = control_qubits
        projected = project_state_onto_bitstring(non_work_state, bitstring)
        np.testing.assert_allclose(projected.probability, 1 / 2**n_ctrl)
        projected.state.specified_qubits = target_qubit
        target_sv = projected.state.get_single_state()

        if bitstring == [True] * n_ctrl:
            # All-ones case: should always flip target (exact behavior guaranteed)
            np.testing.assert_allclose([0, 1], target_sv)
        else:
            # Collect probability for averaging
            prob_target_zero = np.abs(target_sv[0]) ** 2
            target_zero_probs.append(prob_target_zero)

    # Check that on average, target stays |0⟩ with probability 1 - epsilon
    avg_prob_target_zero = np.mean(target_zero_probs)
    assert avg_prob_target_zero >= 1 - epsilon, (
        f"Average target |0⟩ probability {avg_prob_target_zero} too low "
        f"(expected ≥ {1 - epsilon})"
    )


@pytest.mark.parametrize("n_ctrl", [2, 3, 4, 5, 6, 7])
def test_cnx_approx_all_ones_activates(n_ctrl: int) -> None:
    """Test that approximate CNX correctly flips target when all controls are |1⟩."""
    epsilon = 0.1
    cnx_approx_fn = cnx_approx(n_ctrl, epsilon)

    k = min(ceil(log2(1 / epsilon)) + 2, n_ctrl)
    n_ancillas = (k - 2) // 2 + (k - 2) % 2 if k > 2 else 0

    @guppy
    @no_type_check
    def main() -> None:
        controls = qarray(n_ctrl)
        for i in range(n_ctrl):
            x(controls[i])

        target = qubit()

        rng = RNG(1)
        cnx_approx_fn(controls, target, rng)

        state_output("target", target)
        state_output("controls", controls)

        rng.discard()
        discard_array(controls)
        discard(target)

    res = main.emulator(n_ctrl + 1 + n_ancillas).run()
    states = Quest.extract_states_dict(res.results[0].entries)
    target_sv = states["target"].get_single_state()

    np.testing.assert_allclose([0, 1], target_sv)
