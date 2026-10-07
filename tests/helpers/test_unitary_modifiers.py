"""Exercise modifier assertions with a minimal custom rotation."""

from typing import no_type_check

import numpy as np
import pytest
from guppylang import guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.angles import angle
from guppylang.std.builtins import array, control, nat
from guppylang.std.quantum import qubit, reset, rz

from guppyalgos.testing import Endianness, assert_unitary_modifiers


@guppy.unitary
class _Rotation:
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
class _WrongDagger:
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
class _WrongControl:
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
class _WrongControlledPhase:
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
    assert_unitary_modifiers(_Rotation, 1, expected_unitary=expected)


@pytest.mark.parametrize(
    ("mode", "circuit"),
    [
        ("Dagger", _WrongDagger),
        ("Control", _WrongControl),
        ("Controlled dagger", _WrongControlledPhase),
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
        assert_unitary_modifiers(_Rotation, 1, expected_unitary=np.eye(2))


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


def test_rejects_nonunitary_forward() -> None:
    """Reject a reset even though each output column has unit norm."""

    @guppy
    @no_type_check
    def circuit(qs: array[qubit, 1]) -> None:
        reset(qs[0])

    with pytest.raises(AssertionError, match="Forward mode"):
        assert_unitary_modifiers(circuit, 1)
