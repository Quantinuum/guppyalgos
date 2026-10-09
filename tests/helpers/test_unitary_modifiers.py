"""Exercise modifier assertions with a minimal custom rotation."""

from typing import no_type_check

import numpy as np
import pytest
from guppylang import guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.angles import angle
from guppylang.std.builtins import array, control, dagger, nat
from guppylang.std.quantum import cx, discard, qubit, reset, rz

from guppyalgos.testing import Endianness, assert_unitary_modifiers


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
        ("Controlled dagger", _rotation_with_wrong_controlled_phase),
    ],
)
def test_rejects_incorrect_custom_modifiers(
    mode: str, circuit: GuppyFunctionDefinition
) -> None:
    """Reject wrong inverses, controls, and active-branch phase errors."""
    with pytest.raises(AssertionError, match=f"{mode} mode"):
        assert_unitary_modifiers(circuit, 1)


def test_rejects_incorrect_forward_reference() -> None:
    """Modifier consistency alone cannot establish algorithm correctness."""
    with pytest.raises(AssertionError, match="Forward mode"):
        assert_unitary_modifiers(
            _rotation_with_correct_implementation, 1, expected_unitary=np.eye(2)
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
        assert_unitary_modifiers(circuit, 1)


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
