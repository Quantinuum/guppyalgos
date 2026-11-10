"""CNX tests."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import no_type_check

import numpy as np
import pytest
from guppylang.decorator import guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.builtins import array, nat, control, dagger
from guppylang.std.debug import state_output
from guppylang.std.quantum import discard, discard_array, h, qubit
from selene_sim import Quest

from guppyalgos.primitives.gate_decompositions.cnx.cnx import (
    _rtof4,
    _tof5,
    cnx,
    cnx_toffoli_ladder,
    cnx_single_ancilla,
)
from guppyalgos.primitives.gate_decompositions.cnx.cnx_cca import (
    cat_cca,
    cnx_cca_logdepth,
    cnx_cca_logdepth_dirty,
)
from guppyalgos.primitives.gate_decompositions.cnx.cnx_teleportation import (
    cnx_teleportation,
    _get_num_ancillas_cnx_teleportation,
)
from guppyalgos.utils import qarray
from guppyalgos.utils import int_to_bits
from guppyalgos.testing import (
    assert_unitary_modifiers,
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


def test_relative_toffoli_modifiers() -> None:
    """Check the optimized inverse and controls against the exact unitary."""

    @guppy(unitary=True)
    @no_type_check
    def circuit(qs: array[qubit, 4]) -> None:
        _rtof4(qs[0], qs[1], qs[2], qs[3])

    # In big-endian |abcd> order, the first 12 basis states are unchanged.
    # |1100> and |1101> acquire +i and -i; |1110> -> -|1111>, |1111> -> |1110>.
    expected = np.eye(16, dtype=np.complex128)
    expected[12:, 12:] = [
        [1j, 0, 0, 0],
        [0, -1j, 0, 0],
        [0, 0, 0, 1],
        [0, 0, -1, 0],
    ]
    assert_unitary_modifiers(circuit, 4, expected_unitary=expected)


def test_tof5_modifier_compilation() -> None:
    """Compile added controls and adjoints for the five-qubit Toffoli helper."""

    @guppy
    @no_type_check
    def circuit(qs: array[qubit, 6], cs: array[qubit, 2]) -> None:
        with control(cs), dagger:
            _tof5(qs[0], qs[1], qs[2], qs[3], qs[4], qs[5])

    circuit.compile_function()


def test_cat_cca_modifier_compilation() -> None:
    """Compile added controls and adjoints for the CNX CCA catalyst."""

    @guppy
    @no_type_check
    def circuit(qs: array[qubit, 3], cs: array[qubit, 2]) -> None:
        with control(cs), dagger:
            cat_cca(qs[0], qs[1], qs[2])

    circuit.compile_function()
