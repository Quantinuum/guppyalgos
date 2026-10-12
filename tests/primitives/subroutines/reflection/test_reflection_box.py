"""Tests for the Reflection box implementation."""

import numpy as np
import pytest

from typing import no_type_check

from guppylang.decorator import guppy
from guppylang.std.builtins import comptime
from guppylang.std.quantum import discard_array, h
from guppylang.std.debug import state_result
from selene_sim import Quest

from guppyalgos.primitives.gate_decompositions.cnx.cnx import cnx
from guppyalgos.primitives.subroutines.reflection.reflection_box import reflection_box
from guppyalgos.utils import transversal, qarray, int_to_bits
from guppyalgos.testing import (
    project_state_onto_bitstring,
    get_total_state_on_only_specified_registers,
)
from tests.primitives.gate_decompositions.cnx.test_cnx import CnxMethod


@pytest.mark.parametrize("n_qubits", [2, 3, 4, 5, 6])
@pytest.mark.parametrize(
    "cnx_method",
    [
        CnxMethod(cnx, lambda n: n - 1),
    ],
)
def test_reflection_box_statevector(n_qubits: int, cnx_method: CnxMethod) -> None:
    r"""Test that reflection flips the phase of only the all-zero control state.

    Verifies that the target statevector for the all-zero control bitstring differs
    from all other control bitstrings by a global sign, confirming the reflection
    acts as :math:`-I` on :math:`\lvert 0^n\rangle` and as the identity on all
    other basis states.

    """
    n_ancillas = cnx_method.ancilla_func(n_qubits - 1)
    cnxmethod = cnx_method.cnx_func
    n_controls = n_qubits - 1

    @guppy
    @no_type_check
    def main() -> None:
        qreg = qarray(comptime(n_qubits))

        # Apply Hadamard gates to all control qubits
        # to create a superposition of all control states.
        for i in range(comptime(n_controls)):
            h(qreg[i])

        reflection_box(qreg, cnxmethod[comptime(n_controls)])

        state_result("qreg", qreg)
        discard_array(qreg)

    res = main.emulator(n_qubits + n_ancillas).run()
    states = Quest.extract_states_dict(res.results[0].entries)

    non_work_state, spec_qubit_dict = get_total_state_on_only_specified_registers(
        states, ["qreg"]
    )
    control_qubits = spec_qubit_dict["qreg"][:-1]
    control_bitstrings = [int_to_bits(i, n_controls) for i in range(2**n_controls)]
    statevectors = {}
    for bitstring in control_bitstrings:
        non_work_state.specified_qubits = control_qubits
        projected = project_state_onto_bitstring(
            non_work_state, list(reversed(bitstring))
        )
        statevectors[tuple(bitstring)] = projected.state.state
        np.testing.assert_allclose(projected.probability, 1 / 2**n_controls)
    all_0 = tuple([False] * n_controls)
    sv_all_0 = statevectors[all_0]
    for bitstring, sv in statevectors.items():
        sign = 1 if bitstring == all_0 else -1
        np.testing.assert_allclose(sv, sign * sv_all_0)


@pytest.mark.parametrize("n_qubits", [2, 3, 4, 5, 6])
@pytest.mark.parametrize(
    "cnx_method",
    [
        CnxMethod(cnx, lambda n: n - 1),
    ],
)
def test_conjugated_reflection_box_statevector(
    n_qubits: int, cnx_method: CnxMethod
) -> None:
    r"""Test that the :math:`H`-conjugated reflection is an open-controlled X gate.

    The target qubit is flipped from :math:`\lvert 0\rangle` to
    :math:`\lvert 1\rangle` only when every control qubit is in state
    :math:`\lvert 0\rangle`. It is left unchanged for all other control states.

    """
    n_ancillas = cnx_method.ancilla_func(n_qubits - 1)
    cnxmethod = cnx_method.cnx_func
    n_controls = n_qubits - 1

    @guppy
    @no_type_check
    def main() -> None:
        qreg = qarray(comptime(n_qubits))

        transversal(h, qreg)

        reflection_box(qreg, cnxmethod[comptime(n_controls)])

        idx = comptime(n_controls)
        h(qreg[idx])

        state_result("qreg", qreg)

        discard_array(qreg)

    res = main.emulator(n_qubits + n_ancillas).run()
    states = Quest.extract_states_dict(res.results[0].entries)

    non_work_state, spec_qubit_dict = get_total_state_on_only_specified_registers(
        states, ["qreg"]
    )
    control_qubits = spec_qubit_dict["qreg"][:-1]
    target_qubit = [spec_qubit_dict["qreg"][-1]]
    control_bitstrings = [int_to_bits(i, n_controls) for i in range(2**n_controls)]
    for bitstring in control_bitstrings:
        non_work_state.specified_qubits = control_qubits
        projected = project_state_onto_bitstring(
            non_work_state, list(reversed(bitstring))
        )
        np.testing.assert_allclose(projected.probability, 1 / 2**n_controls)
        projected.state.specified_qubits = target_qubit
        target_sv = projected.state.state
        if bitstring == [False] * n_controls:
            np.testing.assert_allclose(np.abs(target_sv), [0, 1])
        else:
            np.testing.assert_allclose(np.abs(target_sv), [1, 0])


@pytest.mark.parametrize(
    ("n_qubits", "n_controls"),
    [
        (2, 0),
        (3, 1),
        (4, 2),
        (5, 3),
        (6, 4),
    ],
)
@pytest.mark.parametrize(
    "cnx_method",
    [
        CnxMethod(cnx, lambda n: n - 1),
    ],
)
def test_reflection_box_invalid_n_qubits(
    n_qubits: int, n_controls: int, cnx_method: CnxMethod
) -> None:
    """Test that ``reflection_box`` raises ``ValueError`` for invalid ``n_qubits``."""
    cnxmethod = cnx_method.cnx_func

    @guppy
    @no_type_check
    def main() -> None:
        qreg = qarray(comptime(n_qubits))
        reflection_box(qreg, cnxmethod[comptime(n_controls)])
        discard_array(qreg)

    with pytest.raises(
        ValueError, match="reflection_box requires n_qubits == n_controls \\+ 1"
    ):
        main.emulator(n_qubits).run()
