"""CNX tests."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import no_type_check

import numpy as np
import pytest
from guppylang.decorator import guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.builtins import array, control, dagger, nat
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
from guppyalgos.utils import _unsafe_array_borrow, _unsafe_array_unborrow, qarray
from guppyalgos.utils import int_to_bits
from guppyalgos.utils.guppy.array import join_arrays, split_array
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


UNITARY_CNX_METHODS = [
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
]


@pytest.mark.parametrize(
    "cnx_method",
    [*UNITARY_CNX_METHODS, CnxMethod(cnx_cca_logdepth_dirty_wrapper, lambda _: 2)],
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

    # get the global phase to check all branches have the same phase
    non_work_state.specified_qubits = control_qubits
    projected = project_state_onto_bitstring(non_work_state, [False] * n_controls)
    projected.state.specified_qubits = target_qubit
    global_phase = projected.state.state[0]

    for bitstring in control_bitstrings:
        non_work_state.specified_qubits = control_qubits
        projected = project_state_onto_bitstring(non_work_state, bitstring)
        np.testing.assert_allclose(projected.probability, 1 / 2**n_controls)
        projected.state.specified_qubits = target_qubit
        target_sv = projected.state.state
        if bitstring == [True] * n_controls:
            np.testing.assert_allclose(global_phase * np.array([0, 1]), target_sv)
        else:
            np.testing.assert_allclose(global_phase * np.array([1, 0]), target_sv)


@pytest.mark.parametrize("cnx_method", UNITARY_CNX_METHODS)
@pytest.mark.parametrize("n_controls", [1, 3, 5])
@pytest.mark.parametrize(
    ("modifier", "n_extra_controls"),
    [
        ("dagger", 0),
        ("controlled", 1),
        ("controlled", 2),
        ("controlled_dagger", 1),
        ("controlled_dagger", 2),
    ],
)
def test_cnx_unitary_modifiers(
    cnx_method: CnxMethod, n_controls: int, modifier: str, n_extra_controls: int
) -> None:
    """Wrap cnx dagger and control which are also cnx and pass them to `test_cnx`."""
    cnx = cnx_method.cnx_func
    n_total_controls = n_controls + n_extra_controls

    @guppy
    @no_type_check
    def daggered(controls: array[qubit, n_total_controls], target: qubit) -> None:
        with dagger:
            cnx(controls, target)

    @guppy
    @no_type_check
    def controlled(controls: array[qubit, n_total_controls], target: qubit) -> None:
        borrowed = _unsafe_array_borrow(controls)
        original, extra = split_array(borrowed, n_controls, n_extra_controls)
        with control(extra):
            cnx(original, target)
        borrowed = join_arrays(original, extra, n_total_controls)
        _unsafe_array_unborrow(controls, borrowed)

    @guppy
    @no_type_check
    def controlled_dagger(
        controls: array[qubit, n_total_controls], target: qubit
    ) -> None:
        borrowed = _unsafe_array_borrow(controls)
        original, extra = split_array(borrowed, n_controls, n_extra_controls)
        with control(extra), dagger:
            cnx(original, target)
        borrowed = join_arrays(original, extra, n_total_controls)
        _unsafe_array_unborrow(controls, borrowed)

    variant = {
        "dagger": daggered,
        "controlled": controlled,
        "controlled_dagger": controlled_dagger,
    }[modifier]
    test_cnx(CnxMethod(variant, cnx_method.ancilla_func), n_total_controls + 1)
