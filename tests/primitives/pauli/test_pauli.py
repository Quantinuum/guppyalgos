"""Tests for pauli utils."""

import numpy as np
from collections.abc import Iterable
import zixy.qubit.pauli as zqp
from guppylang import guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.angles import angle
from guppylang.std.builtins import array, nat
from guppylang.std.debug import state_output
from guppylang.std.quantum import cx, cy, cz, discard, discard_array, h, qubit, x, y, z
from selene_sim import Quest

from guppyalgos.utils import apply_phase, qarray
from guppyalgos.primitives.pauli import pauli_to_cntrl_gate, pauli_to_gate
from guppyalgos.testing import (
    assert_allclose_ignorephase,
    get_statevector,
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


def test_pauli_to_cntrl_gate() -> None:
    """Test pauli_to_cntrl_gate produces the same effect as manual."""
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
        p: GuppyFunctionDefinition[[qubit, array[qubit, n_q]], None],
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
            p(c, q)
            pmanual(c, q)
            state_output("state", c)
            discard_array(q)
            discard(c)

        return main

    # test manual is dagger of pauli_to_cntrl_gate
    for i, pauli in enumerate(pl):
        p = pauli_to_cntrl_gate(pauli, 4)
        pmanual = manual_paulis[i]
        main = make_main(p, pmanual)
        res = main.emulator(n_qubits=5).run()
        states = Quest.extract_states_dict(res.results[0].entries)
        projstate = project_state_onto_bitstring(states["state"], [True])
        assert_allclose_ignorephase(projstate.state.state, zero_state_vec)

    # test when the control qubit is 0
    for i, pauli in enumerate(pl):
        p = pauli_to_cntrl_gate(pauli, 4)
        pmanual = manual_paulis[i]
        main = make_main(p, pmanual, False)
        res = main.emulator(n_qubits=5).run()
        states = Quest.extract_states_dict(res.results[0].entries)
        projstate = project_state_onto_bitstring(states["state"], [False])
        assert_allclose_ignorephase(projstate.state.state, zero_state_vec)


def test_pauli_to_cntrl_gate_applies_phase_once() -> None:
    """Apply one coefficient phase to a controlled Pauli string."""
    coefficient_phase = 0.37
    controlled_pauli = pauli_to_cntrl_gate(
        zqp.String.from_str("X0 Z1", 2), 2, coefficient_phase
    )

    @guppy
    @no_type_check
    def cntrl_pauli_with_phase() -> None:
        control = qubit()
        targets = qarray(2)
        h(control)
        controlled_pauli(control, targets)
        state_output("result_state", control, targets[0], targets[1])
        discard(control)
        discard_array(targets)

    @guppy
    @no_type_check
    def manual_cntrl_pauli_with_phase() -> None:
        control = qubit()
        targets = qarray(2)
        h(control)
        cx(control, targets[0])
        cz(control, targets[1])
        apply_phase(control, angle(coefficient_phase))
        state_output("result_state", control, targets[0], targets[1])
        discard(control)
        discard_array(targets)

    np.testing.assert_allclose(
        get_statevector(cntrl_pauli_with_phase, 3),
        get_statevector(manual_cntrl_pauli_with_phase, 3),
        atol=1e-12,
    )
