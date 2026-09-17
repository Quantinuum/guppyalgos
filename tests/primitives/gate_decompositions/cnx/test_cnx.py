"""CNX tests."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import no_type_check

import numpy as np
import pytest
from guppylang.decorator import guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.builtins import array, nat
from guppylang.std.debug import state_output
from guppylang.std.quantum import discard, discard_array, h, qubit
from selene_sim import Quest

from guppyalgos.primitives.gate_decompositions.cnx.cnx import (
    cnx,
    cnx_toffoli_ladder,
    cnx_single_ancilla,
)
from guppyalgos.primitives.gate_decompositions.cnx.cnx_cca import (
    cnx_cca_logdepth,
    cnx_cca_logdepth_dirty,
)
from guppyalgos.primitives.gate_decompositions.cnx.cnx_teleportation import (
    cnx_teleportation,
    _get_num_ancillas_cnx_teleportation,
)
from guppyalgos.utils import int_to_bits, qarray
from guppyalgos.testing import (
    get_total_state_on_only_specified_registers,
    project_state_onto_bitstring,
)


@dataclass
class CnxMethod[n_controls: nat]:
    """Wrapper for cnx method including ancilla calculation."""

    cnx_func: GuppyFunctionDefinition[[array[qubit, n_controls], qubit], None]
    ancilla_func: Callable[[int], int]


@guppy
@no_type_check
def cnx_cca_logdepth_dirty_wrapper[n_controls: nat](
    controls: array[qubit, n_controls],
    target: qubit,
) -> None:
    """Call cnx_cca_logdepth_dirty with borrowed ancillas in a dirty state."""
    borrowed_a = qubit()
    borrowed_b = qubit()
    h(borrowed_a)
    h(borrowed_b)
    cnx_cca_logdepth_dirty(controls, target, borrowed_a, borrowed_b)
    discard(borrowed_a)
    discard(borrowed_b)


@pytest.mark.parametrize(
    "cnx_method",
    [
        CnxMethod(cnx, lambda n_controls: (n_controls - 2) // 2 + (n_controls - 2) % 2),
        CnxMethod(cnx_toffoli_ladder, lambda n_controls: max(0, n_controls - 2)),
        CnxMethod(cnx_single_ancilla, lambda _: 1),
        CnxMethod(
            cnx_teleportation,
            lambda n_controls: _get_num_ancillas_cnx_teleportation(n_controls),
        ),
        CnxMethod(
            cnx_cca_logdepth,
            lambda n_controls: 1 if n_controls < 6 else 2,
        ),
        CnxMethod(cnx_cca_logdepth_dirty_wrapper, lambda _: 2),
    ],
)
@pytest.mark.parametrize("n_qubits", [2, 3, 4, 5, 6, 7, 8, 9, 10])
def test_cnx(
    cnx_method: CnxMethod,
    n_qubits: int,
) -> None:
    """Test cnx circuit."""
    n_controls = n_qubits - 1
    cnx = cnx_method.cnx_func
    n_ancillas = cnx_method.ancilla_func(n_controls)

    @guppy
    @no_type_check
    def main() -> None:
        target = qubit()
        control_register = qarray(n_controls)

        for c in range(n_controls):
            h(control_register[c])
        cnx(control_register, target)
        state_output("target", target)
        state_output("controls", control_register)
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
        projected = project_state_onto_bitstring(non_work_state, bitstring)
        np.testing.assert_allclose(projected.probability, 1 / 2**n_controls)
        projected.state.specified_qubits = target_qubit
        target_sv = projected.state.get_single_state()
        if bitstring == [True] * n_controls:
            np.testing.assert_allclose([0, 1], target_sv)
        else:
            np.testing.assert_allclose([1, 0], target_sv)
