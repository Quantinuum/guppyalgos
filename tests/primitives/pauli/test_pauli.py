"""Tests for pauli utils."""

import numpy as np
import pytest
from collections.abc import Iterable
import zixy.qubit.pauli as zqp
from guppylang import guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.builtins import array, control, nat
from guppylang.std.debug import state_output
from guppylang.std.quantum import cx, cy, cz, discard, discard_array, qubit, x, y, z
from selene_sim import Quest

from guppyalgos.utils import qarray
from guppyalgos.primitives.pauli import (
    pauli_to_gate,
    pauli_to_z_basis,
)
from guppyalgos.testing import (
    assert_allclose_ignorephase,
    assert_unitary_modifiers,
    project_state_onto_bitstring,
)


from typing import cast, no_type_check


def test_pauli_to_gate() -> None:
    """Test pauli_to__gate produces the same effect as manual."""
    inp_string = "X0 X1 X3, X1 Y2 Z3, X3, Y2 Z3"
    pl = cast(Iterable[zqp.String], zqp.Strings.from_str(inp_string, 4))

    @guppy
    @no_type_check
    def p0manual(q: array[qubit, 4]) -> None:
        x(q[0])
        x(q[1])
        x(q[3])

    @guppy
    @no_type_check
    def p1manual(q: array[qubit, 4]) -> None:
        x(q[1])
        y(q[2])
        z(q[3])

    @guppy
    @no_type_check
    def p2manual(q: array[qubit, 4]) -> None:
        x(q[3])

    @guppy
    @no_type_check
    def p3manual(q: array[qubit, 4]) -> None:
        y(q[2])
        z(q[3])

    manual_paulis = [p0manual, p1manual, p2manual, p3manual]
    zero_state_vec = np.zeros(2**4, dtype=np.complex128)
    zero_state_vec[0] = 1

    def make_main(
        p: GuppyFunctionDefinition[array[qubit, 4], None],
        pmanual: GuppyFunctionDefinition[array[qubit, 4], None],
    ) -> GuppyFunctionDefinition[[], None]:
        @guppy
        @no_type_check
        def main() -> None:
            q = qarray(4)
            p(q)
            pmanual(q)
            state_output("state", q)
            discard_array(q)

        return main

    for i, pauli in enumerate(pl):
        p = pauli_to_gate(pauli, 4)
        pmanual = manual_paulis[i]
        main = make_main(p, pmanual)
        res = main.emulator(n_qubits=4).run()
        states = Quest.extract_states_dict(res.results[0].entries)
        assert_allclose_ignorephase(states["state"].state, zero_state_vec)


def test_controlled_pauli_gate() -> None:
    """Check the public control modifier against manual controlled gates."""
    inp_string = "X0 X1 X3, X1 Y2 Z3, X3, Y2 Z3"
    pl = cast(Iterable[zqp.String], zqp.Strings.from_str(inp_string, 4))

    @guppy
    @no_type_check
    def p0manual(c: qubit, q: array[qubit, 4]) -> None:
        cx(c, q[0])
        cx(c, q[1])
        cx(c, q[3])

    @guppy
    @no_type_check
    def p1manual(c: qubit, q: array[qubit, 4]) -> None:
        cx(c, q[1])
        cy(c, q[2])
        cz(c, q[3])

    @guppy
    @no_type_check
    def p2manual(c: qubit, q: array[qubit, 4]) -> None:
        cx(c, q[3])

    @guppy
    @no_type_check
    def p3manual(c: qubit, q: array[qubit, 4]) -> None:
        cy(c, q[2])
        cz(c, q[3])

    manual_paulis = [p0manual, p1manual, p2manual, p3manual]
    zero_state_vec = np.zeros(2**4, dtype=np.complex128)
    zero_state_vec[0] = 1

    def make_main[n_q: nat](
        p: GuppyFunctionDefinition[[array[qubit, n_q]], None],
        pmanual: GuppyFunctionDefinition[[qubit, array[qubit, n_q]], None],
        turn_on_control: bool = True,
    ) -> GuppyFunctionDefinition[[], None]:
        @guppy
        @no_type_check
        def main() -> None:
            c = qubit()
            if turn_on_control:
                x(c)
            q = qarray(4)
            with control(c):
                p(q)
            pmanual(c, q)
            state_output("state", c)
            discard_array(q)
            discard(c)

        return main

    # The manual gates undo the controlled Pauli.
    for i, pauli in enumerate(pl):
        p = pauli_to_gate(pauli, 4)
        pmanual = manual_paulis[i]
        main = make_main(p, pmanual)
        res = main.emulator(n_qubits=5).run()
        states = Quest.extract_states_dict(res.results[0].entries)
        projstate = project_state_onto_bitstring(states["state"], [True])
        assert_allclose_ignorephase(projstate.state.state, zero_state_vec)

    # test when the control qubit is 0
    for i, pauli in enumerate(pl):
        p = pauli_to_gate(pauli, 4)
        pmanual = manual_paulis[i]
        main = make_main(p, pmanual, False)
        res = main.emulator(n_qubits=5).run()
        states = Quest.extract_states_dict(res.results[0].entries)
        projstate = project_state_onto_bitstring(states["state"], [False])
        assert_allclose_ignorephase(projstate.state.state, zero_state_vec)


@pytest.mark.parametrize("paulis", ["X0 Y1", "Y0 I1", "Z0 I1"])
def test_pauli_to_z_basis_modifiers(paulis: str) -> None:
    """Check basis changes and all modifiers, including empty X/Y supports."""
    hadamard = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
    matrices = {
        "I": np.eye(2, dtype=np.complex128),
        "X": hadamard,
        "Y": hadamard @ np.diag([1, -1j]),
        "Z": np.eye(2, dtype=np.complex128),
    }
    pauli_string = zqp.String.from_str(paulis, 2)
    first, second = paulis.split()
    expected = np.kron(matrices[first[0]], matrices[second[0]])
    assert_unitary_modifiers(
        pauli_to_z_basis(pauli_string, 2), 2, expected_unitary=expected
    )


def test_composed_pauli_to_z_basis_modifiers() -> None:
    """Check composed modifiers for equal-size supports on different qubits."""
    xy_basis = pauli_to_z_basis(zqp.String.from_str("X0 Y1"), 2)
    yx_basis = pauli_to_z_basis(zqp.String.from_str("Y0 X1"), 2)

    @guppy(unitary=True)
    @no_type_check
    def change_basis(q: array[qubit, 2]) -> None:
        xy_basis(q)
        yx_basis(q)

    hadamard = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    y_to_z = hadamard @ np.diag([1, -1j])
    expected = np.kron(y_to_z @ hadamard, hadamard @ y_to_z)
    assert_unitary_modifiers(change_basis, 2, expected_unitary=expected)
