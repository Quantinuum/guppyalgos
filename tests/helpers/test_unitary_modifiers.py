"""Exercise modifier assertions with a minimal custom rotation."""

from typing import no_type_check

import numpy as np
import pytest
from guppylang import guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.angles import angle
from guppylang.std.builtins import array, control, dagger, nat
from guppylang.std.quantum import cx, discard, h, qubit, reset, ry, rz, x

from guppyalgos.testing import (
    Endianness,
    assert_allclose_ignorephase,
    assert_unitary_modifiers,
    get_unitary,
    switch_matrix_endianness,
)


@guppy.unitary
class _rotation_with_correct_implementation:
    @guppy
    @no_type_check
    def __call__(qs: array[qubit, 1]) -> None:
        rz(qs[0], angle(0.7))

    @guppy
    @no_type_check
    def daggered(qs: array[qubit, 1]) -> None:
        rz(qs[0], angle(-0.7))

    @guppy
    @no_type_check
    def controlled[n: nat](qs: array[qubit, 1], cs: array[qubit, n]) -> None:
        with control(cs):
            rz(qs[0], angle(0.7))

    @guppy
    @no_type_check
    def ctrl_daggered[n: nat](qs: array[qubit, 1], cs: array[qubit, n]) -> None:
        with control(cs):
            rz(qs[0], angle(-0.7))


@guppy.unitary
class _rotation_with_wrong_dagger:
    @guppy
    @no_type_check
    def __call__(qs: array[qubit, 1]) -> None:
        rz(qs[0], angle(0.7))

    @guppy
    @no_type_check
    def daggered(qs: array[qubit, 1]) -> None:
        rz(qs[0], angle(0.7))

    @guppy
    @no_type_check
    def controlled[n: nat](qs: array[qubit, 1], cs: array[qubit, n]) -> None:
        with control(cs):
            rz(qs[0], angle(0.7))

    @guppy
    @no_type_check
    def ctrl_daggered[n: nat](qs: array[qubit, 1], cs: array[qubit, n]) -> None:
        with control(cs):
            rz(qs[0], angle(-0.7))


@guppy.unitary
class _rotation_with_wrong_control:
    @guppy
    @no_type_check
    def __call__(qs: array[qubit, 1]) -> None:
        rz(qs[0], angle(0.7))

    @guppy
    @no_type_check
    def daggered(qs: array[qubit, 1]) -> None:
        rz(qs[0], angle(-0.7))

    @guppy
    @no_type_check
    def controlled[n: nat](qs: array[qubit, 1], cs: array[qubit, n]) -> None:
        with control(cs):
            rz(qs[0], angle(-0.7))

    @guppy
    @no_type_check
    def ctrl_daggered[n: nat](qs: array[qubit, 1], cs: array[qubit, n]) -> None:
        with control(cs):
            rz(qs[0], angle(-0.7))


@guppy.unitary
class _rotation_with_wrong_controlled_phase:
    @guppy
    @no_type_check
    def __call__(qs: array[qubit, 1]) -> None:
        rz(qs[0], angle(0.7))

    @guppy
    @no_type_check
    def daggered(qs: array[qubit, 1]) -> None:
        rz(qs[0], angle(-0.7))

    @guppy
    @no_type_check
    def controlled[n: nat](qs: array[qubit, 1], cs: array[qubit, n]) -> None:
        with control(cs):
            rz(qs[0], angle(0.7))

    @guppy
    @no_type_check
    def ctrl_daggered[n: nat](qs: array[qubit, 1], cs: array[qubit, n]) -> None:
        with control(cs):
            # Two half-turns change only the active branch's sign.
            rz(qs[0], angle(-0.7 + 2.0))


def test_custom_unitary_modifiers() -> None:
    """One invocation validates all four custom implementations."""
    expected = np.diag(np.exp(np.pi * np.array([-0.35j, 0.35j])))
    assert_unitary_modifiers(
        _rotation_with_correct_implementation, 1, expected_unitary=expected
    )


@pytest.mark.parametrize(
    ("mode", "circuit"),
    [
        ("Dagger", _rotation_with_wrong_dagger),
        ("Control", _rotation_with_wrong_control),
    ],
)
def test_rejects_incorrect_custom_modifiers(
    mode: str, circuit: GuppyFunctionDefinition
) -> None:
    """Reject incorrect target operations in dagger and control modes."""
    with pytest.raises(
        AssertionError,
        match=f"{mode} mode:",
    ):
        assert_unitary_modifiers(
            circuit,
            1,
            expected_unitary=np.diag(np.exp(np.pi * np.array([-0.35j, 0.35j]))),
        )


def test_forward_correctness_separately() -> None:
    """Check the intended unitary separately from modifier consistency."""
    expected = np.diag(np.exp(np.pi * np.array([-0.35j, 0.35j])))
    assert_allclose_ignorephase(
        get_unitary(_rotation_with_correct_implementation, 1), expected
    )


def test_rejects_wrong_controlled_dagger_phase() -> None:
    """Coherent checks reject a phase error confined to the active branch."""
    expected = np.diag(np.exp(np.pi * np.array([-0.35j, 0.35j])))
    with pytest.raises(AssertionError, match="Controlled dagger mode:"):
        assert_unitary_modifiers(
            _rotation_with_wrong_controlled_phase, 1, expected_unitary=expected
        )


@pytest.mark.parametrize("endianness", list(Endianness))
def test_automatic_modifiers(endianness: Endianness) -> None:
    """Also support ordinary unitary functions and asymmetric registers."""

    @guppy(unitary=True)
    @no_type_check
    def circuit(qs: array[qubit, 2]) -> None:
        rz(qs[0], angle(0.7))

    expected = np.diag(np.exp(np.pi * np.array([-0.35j, 0.35j])))
    expected = (
        np.kron(expected, np.eye(2))
        if endianness == Endianness.BIG
        else np.kron(np.eye(2), expected)
    )
    assert_unitary_modifiers(
        circuit, 2, expected_unitary=expected, endianness=endianness
    )


def test_bad_reset() -> None:
    """Reject a reset even though each output column has unit norm."""

    @guppy
    @no_type_check
    def circuit(qs: array[qubit, 1]) -> None:
        reset(qs[0])

    with pytest.raises(
        AssertionError,
        match=r"Forward mode: extracted matrix is not unitary.*specialized tests",
    ):
        assert_unitary_modifiers(
            circuit,
            1,
            expected_unitary=np.eye(2),
        )


@guppy
@no_type_check
def _uncompute_and_reset_work(qs: array[qubit, 1]) -> None:
    """Uncompute an entangled work qubit before resetting and discarding it."""
    work = qubit()
    cx(qs[0], work)
    cx(qs[0], work)
    reset(work)
    discard(work)


@guppy.unitary
class _rotation_with_good_reset:
    @guppy
    @no_type_check
    def __call__(qs: array[qubit, 1]) -> None:
        _uncompute_and_reset_work(qs)
        _rotation_with_correct_implementation(qs)

    @guppy
    @no_type_check
    def daggered(qs: array[qubit, 1]) -> None:
        _uncompute_and_reset_work(qs)
        with dagger:
            _rotation_with_correct_implementation(qs)

    @guppy
    @no_type_check
    def controlled[n: nat](qs: array[qubit, 1], cs: array[qubit, n]) -> None:
        _uncompute_and_reset_work(qs)
        with control(cs):
            _rotation_with_correct_implementation(qs)

    @guppy
    @no_type_check
    def ctrl_daggered[n: nat](qs: array[qubit, 1], cs: array[qubit, n]) -> None:
        _uncompute_and_reset_work(qs)
        with control(cs):
            with dagger:
                _rotation_with_correct_implementation(qs)


def test_good_reset() -> None:
    """Accept reset of correctly uncomputed work qubits in all four modes."""
    expected = np.diag(np.exp(np.pi * np.array([-0.35j, 0.35j])))
    assert_unitary_modifiers(
        _rotation_with_good_reset, 1, expected_unitary=expected, n_extra_qubits=1
    )


@guppy.unitary
class _rotation_with_bad_reset:
    @guppy
    @no_type_check
    def __call__(qs: array[qubit, 1], fault_mode: int) -> None:
        rz(qs[0], angle(0.7))

    @guppy
    @no_type_check
    def daggered(qs: array[qubit, 1], fault_mode: int) -> None:
        if fault_mode == 1:
            reset(qs[0])
        rz(qs[0], angle(-0.7))

    @guppy
    @no_type_check
    def controlled[n: nat](
        qs: array[qubit, 1], fault_mode: int, cs: array[qubit, n]
    ) -> None:
        if fault_mode == 2:
            reset(qs[0])
        with control(cs):
            rz(qs[0], angle(0.7))

    @guppy
    @no_type_check
    def ctrl_daggered[n: nat](
        qs: array[qubit, 1], fault_mode: int, cs: array[qubit, n]
    ) -> None:
        if fault_mode == 3:
            reset(qs[0])
        with control(cs):
            rz(qs[0], angle(-0.7))


@pytest.mark.parametrize(
    ("fault_mode", "mode"),
    [(1, "Dagger"), (2, "Control"), (3, "Controlled dagger")],
)
def test_nonunitary_modifier(fault_mode: int, mode: str) -> None:
    """Reject a reset in any modifier, identifying the failing mode."""

    @guppy(unitary=True)
    @no_type_check
    def circuit(qs: array[qubit, 1]) -> None:
        _rotation_with_bad_reset(qs, fault_mode)

    with pytest.raises(AssertionError, match=f"{mode} mode:"):
        assert_unitary_modifiers(
            circuit,
            1,
            expected_unitary=np.diag(np.exp(np.pi * np.array([-0.35j, 0.35j]))),
        )


def test_native_h_modifiers() -> None:
    """A reference resolves H's extracted forward global phase."""

    @guppy(unitary=True)
    @no_type_check
    def circuit(qs: array[qubit, 1]) -> None:
        h(qs[0])

    expected = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
    assert_unitary_modifiers(circuit, 1, expected_unitary=expected)


@guppy.unitary
class _rotation_with_incorrect_projected_block:
    @guppy
    @no_type_check
    def __call__(qs: array[qubit, 1], fault: int) -> None:
        _rotation_with_correct_implementation(qs)

    @guppy
    @no_type_check
    def daggered(qs: array[qubit, 1], fault: int) -> None:
        with dagger:
            _rotation_with_correct_implementation(qs)

    @guppy
    @no_type_check
    def controlled[n: nat](
        qs: array[qubit, 1], fault: int, cs: array[qubit, n]
    ) -> None:
        if fault == 1:
            # Incorrectly act on the target when the control is off too.
            _rotation_with_correct_implementation(qs)
        else:
            with control(cs):
                _rotation_with_correct_implementation(qs)

    @guppy
    @no_type_check
    def ctrl_daggered[n: nat](
        qs: array[qubit, 1], fault: int, cs: array[qubit, n]
    ) -> None:
        if fault == 2:
            with control(cs):
                _rotation_with_correct_implementation(qs)
        else:
            with control(cs), dagger:
                _rotation_with_correct_implementation(qs)


@pytest.mark.parametrize(("fault", "mode"), [(1, "Control"), (2, "Controlled dagger")])
def test_rejects_incorrect_projected_blocks(fault: int, mode: str) -> None:
    """Reject a nonidentity inactive block and a wrong controlled adjoint."""

    @guppy(unitary=True)
    @no_type_check
    def circuit(qs: array[qubit, 1]) -> None:
        _rotation_with_incorrect_projected_block(qs, fault)

    with pytest.raises(
        AssertionError,
        match=f"{mode} mode:",
    ):
        assert_unitary_modifiers(
            circuit,
            1,
            expected_unitary=np.diag(np.exp(np.pi * np.array([-0.35j, 0.35j]))),
        )


def test_rejects_incorrect_forward_reference() -> None:
    """Even consistent modifiers cannot hide an incorrect forward unitary."""
    with pytest.raises(AssertionError, match=r"Forward mode:.*does not match"):
        assert_unitary_modifiers(
            _rotation_with_correct_implementation, 1, expected_unitary=np.eye(2)
        )


def test_reference_phase_is_not_normalized() -> None:
    """Forward phase equivalence must not erase the controlled reference phase."""
    expected = np.exp(0.2j) * np.diag(np.exp(np.pi * np.array([-0.35j, 0.35j])))
    with pytest.raises(AssertionError, match="Control mode:"):
        assert_unitary_modifiers(
            _rotation_with_correct_implementation, 1, expected_unitary=expected
        )


# Rz on the first qubit, an ascending CX chain, then Ry on the last qubit.
@guppy.unitary
class _entangling_rotation_with_custom_modifiers:
    @guppy
    @no_type_check
    def __call__[n_targets: nat](qs: array[qubit, n_targets]) -> None:
        rz(qs[0], angle(0.4))
        for i in range(n_targets - 1):
            cx(qs[i], qs[i + 1])
        ry(qs[n_targets - 1], angle(0.3))

    @guppy
    @no_type_check
    def daggered[n_targets: nat](qs: array[qubit, n_targets]) -> None:
        ry(qs[n_targets - 1], angle(-0.3))
        for offset in range(n_targets - 1):
            i = n_targets - 2 - offset
            cx(qs[i], qs[i + 1])
        rz(qs[0], angle(-0.4))

    @guppy
    @no_type_check
    def controlled[n_targets: nat, n_controls: nat](
        qs: array[qubit, n_targets], cs: array[qubit, n_controls]
    ) -> None:
        with control(cs):
            rz(qs[0], angle(0.4))
        for i in range(n_targets - 1):
            with control(cs):
                cx(qs[i], qs[i + 1])
        with control(cs):
            ry(qs[n_targets - 1], angle(0.3))

    @guppy
    @no_type_check
    def ctrl_daggered[n_targets: nat, n_controls: nat](
        qs: array[qubit, n_targets], cs: array[qubit, n_controls]
    ) -> None:
        with control(cs):
            ry(qs[n_targets - 1], angle(-0.3))
        for offset in range(n_targets - 1):
            i = n_targets - 2 - offset
            with control(cs):
                cx(qs[i], qs[i + 1])
        with control(cs):
            rz(qs[0], angle(-0.4))


@pytest.mark.parametrize("n_qubits", [2, 3])
@pytest.mark.parametrize("endianness", list(Endianness))
def test_multi_qubit_custom_modifiers(n_qubits: int, endianness: Endianness) -> None:
    """Check all custom modifiers of an entangling, non-self-adjoint unitary."""
    dimension = 2**n_qubits
    z_rotation = np.diag(np.exp(np.pi * np.array([-0.2j, 0.2j])))
    c, s = np.cos(0.15 * np.pi), np.sin(0.15 * np.pi)
    y_rotation = np.array([[c, -s], [s, c]], dtype=np.complex128)

    # An ascending CX chain maps each output bit to the input prefix parity.
    ladder = np.zeros((dimension, dimension), dtype=np.complex128)
    for column in range(dimension):
        input_bits = [(column >> i) & 1 for i in reversed(range(n_qubits))]
        output_bits = np.bitwise_xor.accumulate(input_bits)
        row = sum(int(bit) << i for i, bit in enumerate(reversed(output_bits)))
        ladder[row, column] = 1

    expected = (
        np.kron(np.eye(dimension // 2), y_rotation)
        @ ladder
        @ np.kron(z_rotation, np.eye(dimension // 2))
    )
    if endianness == Endianness.LITTLE:
        expected = switch_matrix_endianness(expected)

    @guppy(unitary=True)
    @no_type_check
    def circuit(qs: array[qubit, n_qubits]) -> None:
        _entangling_rotation_with_custom_modifiers(qs)

    assert_unitary_modifiers(
        circuit, n_qubits, expected_unitary=expected, endianness=endianness
    )


@guppy.unitary
class _rotation_with_flipped_control:
    @guppy
    @no_type_check
    def __call__(qs: array[qubit, 1]) -> None:
        _rotation_with_correct_implementation(qs)

    @guppy
    @no_type_check
    def daggered(qs: array[qubit, 1]) -> None:
        with dagger:
            _rotation_with_correct_implementation(qs)

    @guppy
    @no_type_check
    def controlled[n: nat](qs: array[qubit, 1], cs: array[qubit, n]) -> None:
        # X leaves a |+> control unchanged but flips computational basis inputs.
        x(cs[0])
        with control(cs):
            _rotation_with_correct_implementation(qs)

    @guppy
    @no_type_check
    def ctrl_daggered[n: nat](qs: array[qubit, 1], cs: array[qubit, n]) -> None:
        with control(cs), dagger:
            _rotation_with_correct_implementation(qs)


def test_rejects_flipped_control_register() -> None:
    """Cover a control-input error invisible to a single |+> control probe."""
    expected = np.diag(np.exp(np.pi * np.array([-0.35j, 0.35j])))
    with pytest.raises(AssertionError, match=r"Control mode:.*does not match"):
        assert_unitary_modifiers(
            _rotation_with_flipped_control, 1, expected_unitary=expected
        )
