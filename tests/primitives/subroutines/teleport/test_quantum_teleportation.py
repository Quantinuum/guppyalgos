"""Tests for the quantum teleportation."""

from typing import no_type_check

import pytest
from guppylang import guppy
from guppylang.std.debug import state_output
from guppylang.std.quantum import discard_array
from selene_sim import Quest

from guppyalgos.primitives.subroutines.teleport import quantum_teleportation
from guppyalgos.primitives.state_preparation import phase_gradient, ghz_state
from guppyalgos.testing import (
    assert_allclose_ignorephase,
    get_statevector,
    get_total_state_on_only_specified_registers,
)
from guppyalgos.utils import qarray


@pytest.mark.parametrize("n_qubits", [1, 2, 3, 4, 5])
@no_type_check
def test_quantum_teleportation_phase_gradient(n_qubits: int) -> None:
    """Test the quantum teleportation with a phase_gradient state."""
    prep = phase_gradient(n_qubits)

    @guppy
    def main_prep() -> None:
        qs = qarray(n_qubits)
        prep(qs)
        state_output("result_state", qs)
        discard_array(qs)

    @guppy
    def main_qt() -> None:
        qs = qarray(n_qubits)
        out = qarray(n_qubits)

        prep(qs)
        quantum_teleportation(qs, out)
        state_output("out", out)
        discard_array(out)

    simulated_prep = get_statevector(main_prep, n_qubits)

    res = main_qt.emulator(3 * n_qubits).run()
    states = Quest.extract_states_dict(res.results[0].entries)

    out_state, _ = get_total_state_on_only_specified_registers(states, ["out"])

    simulated_qt = out_state.state
    assert_allclose_ignorephase(simulated_prep, simulated_qt)


@pytest.mark.parametrize("n_qubits", [2, 3, 4, 5])
@no_type_check
def test_quantum_teleportation_ghz(n_qubits: int) -> None:
    """Test the quantum teleportation with a ghz state."""

    @guppy
    def main_prep() -> None:
        qs = qarray(n_qubits)
        ghz_state(qs)
        state_output("result_state", qs)
        discard_array(qs)

    @guppy
    def main_qt() -> None:
        qs = qarray(n_qubits)
        out = qarray(n_qubits)

        ghz_state(qs)
        quantum_teleportation(qs, out)
        state_output("out", out)
        discard_array(out)

    simulated_prep = get_statevector(main_prep, n_qubits)

    res = main_qt.emulator(3 * n_qubits).run()
    states = Quest.extract_states_dict(res.results[0].entries)

    out_state, _ = get_total_state_on_only_specified_registers(states, ["out"])

    simulated_qt = out_state.state
    assert_allclose_ignorephase(simulated_prep, simulated_qt)
