"""Qubitized power-oracle behavior and its remaining modifier boundary."""

from typing import no_type_check

import numpy as np
import pytest
from guppylang import guppy
from guppylang.std.builtins import Unitary, array, control, nat
from guppylang.std.quantum import cx, h, qubit, t
from guppylang_internals.checker.errors.generic import InvalidUnderDagger
from guppylang_internals.error import GuppyError

from guppyalgos.algorithms.phase_estimation import (
    QubitizationRegs,
    qubitized_power_oracle,
)
from guppyalgos.testing import Endianness, assert_allclose_ignorephase, get_unitary
from guppyalgos.utils.guppy.unsafe_borrow import (
    _unsafe_array_borrow_slice,
    _unsafe_array_unborrow_slice,
)


@guppy(unitary=True)
@no_type_check
def _controlled_walk(
    ctrl: qubit, prep: array[qubit, 1], target: array[qubit, 1]
) -> None:
    """Use a noncommuting walk with a relative phase on its active branch."""
    with control(ctrl):
        h(prep[0])
        cx(prep[0], target[0])
        t(target[0])


@pytest.mark.parametrize("power", [0, 1, 3])
def test_qubitized_power_oracle_unitary(power: int) -> None:
    """Check all control branches, repetitions, and borrowed-register restoration."""

    @guppy
    @no_type_check
    def circuit(qs: array[qubit, 3]) -> None:
        prefix, prep, target = _unsafe_array_borrow_slice(qs, 1, 1, 1)
        regs = QubitizationRegs(prep, target)
        qubitized_power_oracle(prefix[0], regs, power, _controlled_walk)
        _unsafe_array_unborrow_slice(qs, prefix, regs.prep, regs.target)

    hadamard = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
    cx_matrix = np.eye(4, dtype=np.complex128)[:, [0, 3, 2, 1]]
    target_t = np.diag([1, 1, np.exp(1j * np.pi / 4), np.exp(1j * np.pi / 4)])
    walk = target_t @ cx_matrix @ np.kron(np.eye(2), hadamard)
    expected = np.eye(8, dtype=np.complex128)
    # Little endian: control, preparation, target. Odd indices have control=1.
    expected[1::2, 1::2] = np.linalg.matrix_power(walk, power)
    assert_allclose_ignorephase(
        get_unitary(circuit, 3, endianness=Endianness.LITTLE), expected
    )


def test_qubitized_walk_callback_protocol_inference() -> None:
    """Generic preparation/target sizes no longer block a Unitary walk callback."""

    @guppy(unitary=True)
    @no_type_check
    def apply_walk[n_prep: nat, n_target: nat](
        ctrl: qubit,
        regs: QubitizationRegs[n_prep, array[qubit, n_target]],
        walk: Unitary[[qubit, array[qubit, n_prep], array[qubit, n_target]], None],
    ) -> None:
        walk(ctrl, regs.prep, regs.target)

    @guppy(unitary=True)
    @no_type_check
    def circuit(ctrl: qubit, regs: QubitizationRegs[1, array[qubit, 1]]) -> None:
        apply_walk(ctrl, regs, _controlled_walk)

    circuit.compile_function()


@pytest.mark.xfail(
    strict=True,
    raises=GuppyError,
    reason="Guppy 1.2.0 rejects automatic dagger generation for runtime walk powers",
)
def test_qubitized_power_oracle_runtime_dagger_generation() -> None:
    """Isolate the remaining loop blocker using the power oracle's exact types."""

    @guppy(unitary=True)
    @no_type_check
    def power_oracle[n_prep: nat, n_target: nat](
        ctrl: qubit,
        regs: QubitizationRegs[n_prep, array[qubit, n_target]],
        power: int,
        walk: Unitary[[qubit, array[qubit, n_prep], array[qubit, n_target]], None],
    ) -> None:
        for _ in range(power):
            walk(ctrl, regs.prep, regs.target)

    @guppy(unitary=True)
    @no_type_check
    def circuit(
        ctrl: qubit, regs: QubitizationRegs[1, array[qubit, 1]], power: int
    ) -> None:
        power_oracle(ctrl, regs, power, _controlled_walk)

    try:
        circuit.compile_function()
    except GuppyError as error:
        # Only the documented dagger-loop diagnostic counts as this blocker.
        if not isinstance(error.error, InvalidUnderDagger):
            raise AssertionError(
                "Expected the runtime dagger-loop diagnostic"
            ) from error
        raise
