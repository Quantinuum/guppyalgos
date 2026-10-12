"""Tests for the Hadamard test primitive."""

from __future__ import annotations

from typing import no_type_check

import numpy as np
from guppylang import guppy
from guppylang.std.angles import pi
from guppylang.std.builtins import array, comptime, nat
from guppylang.std.debug import state_result
from guppylang.std.quantum import (
    cz,
    discard,
    discard_array,
    crz,
    h,
    qubit,
    x,
)

from guppyalgos.algorithms.phase_estimation import hadamard_test
from guppyalgos.utils import qarray
from guppyalgos.testing import assert_allclose_ignorephase, get_statevector


@guppy.struct
class UnitaryRegs[n_state: nat]:
    """Simple structured register used to exercise generic register handling."""

    system: array[qubit, n_state]


def test_hadamard_test_phase_kickback_on_eigenstate() -> None:
    """A controlled-Z on a ``|1>`` eigenstate should kick back a ``-1`` phase."""

    @guppy
    @no_type_check
    def cntrl_z(ancilla: qubit, unitary_regs: UnitaryRegs[1]) -> None:
        cz(ancilla, unitary_regs.system[0])

    @guppy
    @no_type_check
    def main() -> None:
        unitary_regs = UnitaryRegs(qarray(comptime(1)))
        ancilla = qubit()

        x(unitary_regs.system[0])
        hadamard_test(ancilla, unitary_regs, cntrl_z)
        state_result("result_state", ancilla, unitary_regs.system[0])
        discard(ancilla)
        discard_array(unitary_regs.system)

    result_state = get_statevector(main, 2)
    expected_state = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.complex128)

    assert_allclose_ignorephase(result_state, expected_state)


def test_hadamard_test_two_crz_phases_add_on_common_eigenstate() -> None:
    """Two commuting CRZ gates should add their phases on a joint eigenstate."""
    phi_1 = 0.25
    phi_2 = 0.5
    expected_phase = np.exp(-1j * np.pi * (phi_1 + phi_2))

    @guppy
    @no_type_check
    def cntrl_two_crz(ancilla: qubit, unitary_regs: UnitaryRegs[2]) -> None:
        crz(ancilla, unitary_regs.system[0], -2 * pi * comptime(phi_1))
        crz(ancilla, unitary_regs.system[1], -2 * pi * comptime(phi_2))

    @guppy
    @no_type_check
    def main() -> None:
        unitary_regs = UnitaryRegs(qarray(comptime(2)))
        ancilla = qubit()

        x(unitary_regs.system[0])
        x(unitary_regs.system[1])
        hadamard_test(ancilla, unitary_regs, cntrl_two_crz)
        state_result(
            "result_state",
            ancilla,
            unitary_regs.system[0],
            unitary_regs.system[1],
        )
        discard(ancilla)
        discard_array(unitary_regs.system)

    result_state = get_statevector(main, 3)
    ancilla_state = (
        np.array([1.0 + expected_phase, 1.0 - expected_phase], dtype=np.complex128) / 2
    )
    expected_state = np.kron(
        ancilla_state, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.complex128)
    )

    assert_allclose_ignorephase(result_state, expected_state)


def test_hadamard_test_bell_state() -> None:
    """The superposition case should produce a Bell state."""

    @guppy
    @no_type_check
    def cntrl_z(ancilla: qubit, main_qubit: qubit) -> None:
        cz(ancilla, main_qubit)

    @guppy
    @no_type_check
    def main() -> None:
        main_qubit = qubit()
        hadamard_anc = qubit()

        h(main_qubit)
        hadamard_test(hadamard_anc, main_qubit, cntrl_z)
        state_result("result_state", hadamard_anc, main_qubit)
        discard(main_qubit)
        discard(hadamard_anc)

    result_state = get_statevector(main, 2)
    expected_state = np.array([1, 0, 0, 1], dtype=np.complex128) / np.sqrt(2)

    assert_allclose_ignorephase(result_state, expected_state)
