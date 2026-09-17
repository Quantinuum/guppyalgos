"""Tests for the controlled reflection box implementation."""

from typing import no_type_check

import numpy as np
import pytest
from guppylang.decorator import guppy
from guppylang.std.builtins import comptime
from guppylang.std.debug import state_result
from guppylang.std.quantum import qubit, discard, discard_array, h
from selene_sim import Quest

from guppyalgos.primitives.gate_decompositions.cnx.cnx import cnx
from guppyalgos.primitives.subroutines.reflection import cntrl_reflection_box
from guppyalgos.utils import int_to_bits, qarray, transversal
from tests.primitives.gate_decompositions.cnx.test_cnx import CnxMethod
from guppyalgos.testing import (
    get_total_state_on_only_specified_registers,
    project_state_onto_bitstring,
)


@pytest.mark.parametrize("n_qubits", [2, 3, 4, 5, 6])
@pytest.mark.parametrize(
    "cnx_method",
    [
        CnxMethod(cnx, lambda n: n - 1),
    ],
)
def test_cntrl_reflection_box_statevector(
    n_qubits: int,
    cnx_method: CnxMethod,
) -> None:
    r"""Test the controlled phase flip using an external-control superposition.

    Prepare the external control in :math:`|+\rangle`, the first
    ``n_qubits - 1`` qubits of ``qreg`` in a uniform superposition, and the
    final ``qreg`` target in :math:`|0\rangle`. On the all-zero ``qreg``
    branch, the controlled reflection negates only the :math:`|1\rangle`
    component of the external control, changing it from :math:`|+\rangle` to
    :math:`|-\rangle`. All other ``qreg`` branches leave it unchanged.

    A final Hadamard maps these relative phases to computational-basis states:
    the external control is :math:`|1\rangle` for the all-zero ``qreg`` branch
    and :math:`|0\rangle` otherwise. Projecting each complete ``qreg``
    bitstring verifies these states and their uniform probabilities. The fixed
    zero appended to each projection also detects any unexpected target-qubit
    leakage.

    """
    n_ancillas = cnx_method.ancilla_func(n_qubits)
    cnxmethod = cnx_method.cnx_func
    n_controls = n_qubits - 1

    @guppy
    @no_type_check
    def main() -> None:
        control_qubit = qubit()
        qreg = qarray(comptime(n_qubits))

        h(control_qubit)
        for i in range(comptime(n_controls)):
            h(qreg[i])

        cntrl_reflection_box(control_qubit, qreg, cnxmethod[comptime(n_qubits)])
        h(control_qubit)

        state_result("external_control", control_qubit)
        state_result("qreg", qreg)
        discard(control_qubit)
        discard_array(qreg)

    result = main.emulator(n_qubits + n_ancillas + 1).run()
    states = Quest.extract_states_dict(result.results[0].entries)

    non_work_state, specified_qubits = get_total_state_on_only_specified_registers(
        states, ["qreg", "external_control"]
    )
    external_control_qubit = specified_qubits["external_control"]
    qreg_qubits = specified_qubits["qreg"]
    control_bitstrings = [int_to_bits(i, n_controls) for i in range(2**n_controls)]
    for bitstring in control_bitstrings:
        non_work_state.specified_qubits = qreg_qubits
        qreg_projected = project_state_onto_bitstring(
            non_work_state,
            [*reversed(bitstring), False],
            new_specified_qubits=external_control_qubit,
        )
        np.testing.assert_allclose(qreg_projected.probability, 1 / 2**n_controls)
        expected_state = np.array([1, 0], dtype=complex)
        if bitstring == [False] * n_controls:
            expected_state = np.array([0, 1], dtype=complex)
        np.testing.assert_allclose(qreg_projected.state.state, expected_state)


@pytest.mark.parametrize("n_qubits", [2, 3, 4, 5, 6])
@pytest.mark.parametrize(
    "cnx_method",
    [
        CnxMethod(cnx, lambda n: n - 1),
    ],
)
def test_conjugated_cntrl_reflection_box_statevector(
    n_qubits: int,
    cnx_method: CnxMethod,
) -> None:
    r"""Test the conjugated reflection by resolving every control branch.

    Prepare the external control and every qubit in ``qreg`` in
    :math:`|+\rangle`. For an active external control and all-zero ``qreg``
    controls, the reflection negates the target's :math:`|0\rangle` component,
    changing :math:`|+\rangle` to :math:`-|-\rangle`. A final Hadamard maps
    this state to :math:`-|1\rangle`. For every other control branch, the
    target remains :math:`|+\rangle` and maps to :math:`|0\rangle`.

    Project the external control and then each little-endian ``qreg`` control
    bitstring. Their probabilities must be uniform, and the remaining target
    state must be ``[0, -1]`` only for the active, all-zero branch and
    ``[1, 0]`` everywhere else. This checks the relative phase as well as the
    branch on which the controlled reflection acts.

    """
    n_ancillas = cnx_method.ancilla_func(n_qubits)
    cnxmethod = cnx_method.cnx_func
    n_controls = n_qubits - 1

    @guppy
    @no_type_check
    def main() -> None:
        control_qubit = qubit()
        qreg = qarray(comptime(n_qubits))

        h(control_qubit)
        transversal(h, qreg)
        cntrl_reflection_box(control_qubit, qreg, cnxmethod[comptime(n_qubits)])
        idx = comptime(n_controls)
        h(qreg[idx])

        state_result("external_control", control_qubit)
        state_result("qreg", qreg)

        discard(control_qubit)
        discard_array(qreg)

    result = main.emulator(n_qubits + n_ancillas + 1).run()
    states = Quest.extract_states_dict(result.results[0].entries)

    non_work_state, specified_qubits = get_total_state_on_only_specified_registers(
        states, ["qreg", "external_control"]
    )
    external_control_qubit = specified_qubits["external_control"]
    qreg_control_qubits = specified_qubits["qreg"][:-1]
    target_qubit = [specified_qubits["qreg"][-1]]
    control_bitstrings = [int_to_bits(i, n_controls) for i in range(2**n_controls)]

    for control_active in [False, True]:
        non_work_state.specified_qubits = external_control_qubit
        external_control_projected = project_state_onto_bitstring(
            non_work_state,
            [control_active],
            new_specified_qubits=qreg_control_qubits,
        )
        np.testing.assert_allclose(external_control_projected.probability, 1 / 2)

        for bitstring in control_bitstrings:
            projected = project_state_onto_bitstring(
                external_control_projected.state,
                list(reversed(bitstring)),
                new_specified_qubits=target_qubit,
            )
            np.testing.assert_allclose(projected.probability, 1 / 2**n_controls)
            expected_state = np.array([1, 0], dtype=complex)
            if control_active and bitstring == [False] * n_controls:
                expected_state = np.array([0, -1], dtype=complex)
            np.testing.assert_allclose(projected.state.state, expected_state)
