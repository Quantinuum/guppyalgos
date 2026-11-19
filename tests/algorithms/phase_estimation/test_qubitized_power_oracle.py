"""Phase-sensitive tests for powered walks and their explicit modifiers."""

from typing import no_type_check

import numpy as np
import pytest
from guppylang import guppy
from guppylang.std.angles import angle
from guppylang.std.builtins import array, control, dagger, nat
from guppylang.std.quantum import qubit, ry, rz

from guppyalgos.algorithms.phase_estimation import qubitized_power_oracle
from guppyalgos.testing import assert_unitary_modifiers
from guppyalgos.utils.guppy.unsafe_borrow import (
    _unsafe_array_borrow_slice,
    _unsafe_array_unborrow_slice,
)


@guppy(unitary=True)
@no_type_check
def _walk[n_prep: nat, n_target: nat](
    intrinsic: qubit, prep: array[qubit, n_prep], target: array[qubit, n_target]
) -> None:
    with control(intrinsic):
        ry(prep[0], angle(0.31))
        with control(prep[0]):
            rz(target[0], angle(0.27))
        rz(prep[0], angle(0.43))


def _walk_matrix() -> np.ndarray:
    """Build Rz(prep) controlled-Rz(target) Ry(prep) independently."""
    theta = np.pi * 0.31 / 2
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    prep_phase = np.diag(np.exp(np.array([-1j, 1j]) * np.pi * 0.43 / 2))
    select = np.diag(
        [1, 1, np.exp(-1j * np.pi * 0.27 / 2), np.exp(1j * np.pi * 0.27 / 2)]
    )
    return np.kron(prep_phase, np.eye(2)) @ select @ np.kron(rotation, np.eye(2))


# Adapt the separate oracle registers to the assertion helper's flat array.
@guppy.unitary
class _apply_power:
    @guppy
    @no_type_check
    def __call__(qs: array[qubit, 3], power: int) -> None:
        intrinsic, prep, target = _unsafe_array_borrow_slice(qs, 1, 1, 1)
        qubitized_power_oracle(intrinsic[0], prep, target, power, _walk)
        _unsafe_array_unborrow_slice(qs, intrinsic, prep, target)

    @guppy
    @no_type_check
    def daggered(qs: array[qubit, 3], power: int) -> None:
        intrinsic, prep, target = _unsafe_array_borrow_slice(qs, 1, 1, 1)
        with dagger:
            qubitized_power_oracle(intrinsic[0], prep, target, power, _walk)
        _unsafe_array_unborrow_slice(qs, intrinsic, prep, target)

    @guppy
    @no_type_check
    def controlled[n: nat](
        qs: array[qubit, 3], power: int, controls: array[qubit, n]
    ) -> None:
        intrinsic, prep, target = _unsafe_array_borrow_slice(qs, 1, 1, 1)
        with control(controls):
            qubitized_power_oracle(intrinsic[0], prep, target, power, _walk)
        _unsafe_array_unborrow_slice(qs, intrinsic, prep, target)

    @guppy
    @no_type_check
    def ctrl_daggered[n: nat](
        qs: array[qubit, 3], power: int, controls: array[qubit, n]
    ) -> None:
        intrinsic, prep, target = _unsafe_array_borrow_slice(qs, 1, 1, 1)
        with control(controls), dagger:
            qubitized_power_oracle(intrinsic[0], prep, target, power, _walk)
        _unsafe_array_unborrow_slice(qs, intrinsic, prep, target)


@pytest.mark.parametrize("power", [-1, 0, 1, 3])
def test_power_oracle_modifiers(power: int) -> None:
    """Check the forward, dagger, controlled, and controlled dagger matrices."""

    @guppy(unitary=True)
    @no_type_check
    def circuit(qs: array[qubit, 3]) -> None:
        _apply_power(qs, power)

    expected = np.eye(8, dtype=complex)
    expected[4:, 4:] = np.linalg.matrix_power(_walk_matrix(), max(power, 0))
    assert_unitary_modifiers(circuit, 3, expected_unitary=expected)
