"""CNZ tests."""

import numpy as np
import pytest

from typing import no_type_check

from guppylang.decorator import guppy
from guppylang.std.builtins import comptime
from guppylang.std.debug import state_result
from guppylang.std.quantum import discard, discard_array, h, qubit
from selene_sim import Quest

from guppyalgos.utils import transversal, qarray, int_to_bits
from guppyalgos.primitives.gate_decompositions.cnx.cnx import cnx_single_ancilla
from guppyalgos.primitives.gate_decompositions.cnx.cnz import cnz
from guppyalgos.testing import (
    get_total_state_on_only_specified_registers,
    project_state_onto_bitstring,
)
from tests.primitives.gate_decompositions.cnx.test_cnx import CnxMethod


@pytest.mark.parametrize(
    "cnx_method",
    [
        CnxMethod(cnx_single_ancilla, lambda _: 1),
    ],
)
@pytest.mark.parametrize("n_qubits", [2, 3, 4, 5, 6, 7, 8, 9, 10])
def test_cnz(
    cnx_method: CnxMethod,
    n_qubits: int,
) -> None:
    """Test cnz circuit."""
    n_controls = n_qubits - 1
    cnx = cnx_method.cnx_func
    n_ancillas = cnx_method.ancilla_func(n_controls)

    @guppy
    @no_type_check
    def main() -> None:
        target = qubit()
        control_register = qarray(comptime(n_controls))

        transversal(h, control_register)
        h(target)
        cnz(control_register, target, cnx)
        h(target)
        state_result("target", target)
        state_result("controls", control_register)
        discard_array(control_register)
        discard(target)

    res = main.emulator(n_qubits + n_ancillas).run()
    states = Quest.extract_states_dict(res.results[0].entries)

    non_work_state, spec_qubit_dict = get_total_state_on_only_specified_registers(
        states, ["target", "controls"]
    )
    control_qubits = spec_qubit_dict["controls"]
    target_qubit = spec_qubit_dict["target"]
    control_bitstrings = [int_to_bits(i, n_controls) for i in range(2**n_controls)]
    for bitstring in control_bitstrings:
        non_work_state.specified_qubits = control_qubits
        projected = project_state_onto_bitstring(
            non_work_state, list(reversed(bitstring))
        )
        np.testing.assert_allclose(projected.probability, 1 / 2**n_controls)
        projected.state.specified_qubits = target_qubit
        target_sv = projected.state.get_single_state()
        if bitstring == [True] * n_controls:
            np.testing.assert_allclose([0, 1], target_sv)
        else:
            np.testing.assert_allclose([1, 0], target_sv)
