"""Test CCU gate decompositions."""

import pytest
import numpy as np
from itertools import product
from guppylang import guppy
from collections.abc import Callable
from guppylang.std.quantum import qubit, x, array, angle
from guppylang.std.builtins import comptime
from guppylang.std.debug import state_output
from guppylang.defs import GuppyFunctionDefinition
from guppyalgos.primitives.gate_decompositions.cnx.ccu import (
    ccx,
    ccy,
    ccz,
    ccry_toffoli,
    ccry_cx,
)
from guppyalgos.testing import get_statevector, get_unitary, assert_allclose_ignorephase


from typing import no_type_check


@guppy
@no_type_check
def guppy_test_program(
    input_state: tuple[int, int, int] @ comptime,
    gate_to_test: Callable[[qubit, qubit, qubit], None],
) -> None:
    """Test program to apply a 3-qubit gate and measure the result.

    Args:
        input_state: The initial state of the qubits.
        gate_to_test: The 3-qubit gate to test.

    """
    control1 = qubit()
    control2 = qubit()
    target = qubit()

    if input_state[0] == 1:
        x(control1)
    if input_state[1] == 1:
        x(control2)
    if input_state[2] == 1:
        x(target)

    gate_to_test(control1, control2, target)

    state_output("result_state", control1, control2, target)

    control1.discard()
    control2.discard()
    target.discard()


@pytest.mark.parametrize(
    ("input_state", "output_state"),
    [
        # tuples are (control_1, control_2, target)
        ((0, 0, 0), (0, 0, 0)),
        ((0, 0, 1), (0, 0, 1)),
        ((1, 0, 0), (1, 0, 0)),
        ((1, 0, 1), (1, 0, 1)),
        ((0, 1, 0), (0, 1, 0)),
        ((0, 1, 1), (0, 1, 1)),
        ((1, 1, 0), (1, 1, 1)),
        ((1, 1, 1), (1, 1, 0)),
    ],
)
def test_ccx_gate(
    input_state: tuple[int, int, int], output_state: tuple[int, int, int]
) -> None:
    """Test CCX gate with different input state and assert the expected output state."""

    # Arrange & Act
    @guppy
    @no_type_check
    def main() -> None:
        guppy_test_program(input_state, ccx)

    result_state = get_statevector(main, 3)
    # Assert
    # The output state is a computational basis state
    # So the amplitude (i.e. probability) at the corresponding index must be 1
    ind = 4 * output_state[2] + 2 * output_state[1] + output_state[0]
    assert np.isclose(np.abs(result_state[ind]), 1.0)


@pytest.mark.parametrize(
    ("input_state", "output_state"),
    [
        # tuples are (control_1, control_2, target)
        ((0, 0, 0), (0, 0, 0)),
        ((0, 0, 1), (0, 0, 1)),
        ((1, 0, 0), (1, 0, 0)),
        ((1, 0, 1), (1, 0, 1)),
        ((0, 1, 0), (0, 1, 0)),
        ((0, 1, 1), (0, 1, 1)),
        ((1, 1, 0), (1, 1, 1)),
        ((1, 1, 1), (1, 1, 0)),
    ],
)
def test_ccy_gate(
    input_state: tuple[int, int, int], output_state: tuple[int, int, int]
) -> None:
    """Test CCY gate with different input state and assert the expected output state."""

    # Arrange & Act
    @guppy
    @no_type_check
    def main() -> None:
        guppy_test_program(input_state, ccy)

    result_state = get_statevector(main, 3)
    # Assert
    # The output state is a computational basis state
    # So the amplitude (i.e. probability) at the corresponding index must be 1
    ind = 4 * output_state[2] + 2 * output_state[1] + output_state[0]
    assert np.isclose(np.abs(result_state[ind]), 1.0)


@pytest.mark.parametrize(
    ("input_state", "output_state"),
    [
        # tuples are (control_1, control_2, target)
        ((0, 0, 0), (0, 0, 0)),
        ((0, 0, 1), (0, 0, 1)),
        ((1, 0, 0), (1, 0, 0)),
        ((1, 0, 1), (1, 0, 1)),
        ((0, 1, 0), (0, 1, 0)),
        ((0, 1, 1), (0, 1, 1)),
        ((1, 1, 0), (1, 1, 0)),
        ((1, 1, 1), (1, 1, 1)),
    ],
)
def test_ccz_gate(
    input_state: tuple[int, int, int], output_state: tuple[int, int, int]
) -> None:
    """Test CCZ gate with different input state and assert the expected output state."""

    # Arrange & Act
    @guppy
    @no_type_check
    def main() -> None:
        guppy_test_program(input_state, ccz)

    result_state = get_statevector(main, 3)
    # Assert
    # The output state is a computational basis state
    # So the amplitude (i.e. probability) at the corresponding index must be 1
    ind = 4 * output_state[2] + 2 * output_state[1] + output_state[0]
    assert np.isclose(np.abs(result_state[ind]), 1.0)


@pytest.mark.parametrize(
    ("input_state", "ccu"),
    list(
        product(
            [
                # tuples are (control_1, control_2, target)
                (1, 1, 0),
                (1, 1, 1),
            ],
            [ccx, ccy, ccz],
        )
    ),
)
def test_ccu_gate_are_self_adjoint(
    input_state: tuple[int, int, int], ccu: Callable[[qubit, qubit, qubit], None]
) -> None:
    """Test double application the same CCU gate."""

    # Arrange & Act
    @guppy
    @no_type_check
    def ccu_ccu(control1: qubit, control2: qubit, target: qubit) -> None:
        # Two application of ccu must be identity
        ccu(control1, control2, target)
        ccu(control1, control2, target)

    @guppy
    @no_type_check
    def main() -> None:
        guppy_test_program(input_state, ccu_ccu)

    result_state = get_statevector(main, 3)
    # Assert
    # We must get the input computational basis state
    # So the amplitude (i.e. probability) at the corresponding index must be 1
    ind = 4 * input_state[2] + 2 * input_state[1] + input_state[0]
    assert np.isclose(np.abs(result_state[ind]), 1.0)


def ccry_unitary(theta: float):
    """Return the exact 8x8 unitary matrix for the CCRY gate."""
    U = np.eye(8, dtype=complex)

    cos_half = np.cos(theta / 2)
    sin_half = np.sin(theta / 2)
    ry = np.array([[cos_half, -sin_half], [sin_half, cos_half]], dtype=complex)

    U[6:8, 6:8] = ry

    return U


@pytest.mark.parametrize("ccry_fun", [ccry_toffoli, ccry_cx])
def test_ccry_gate(
    ccry_fun: GuppyFunctionDefinition[[qubit, qubit, qubit, angle], None],
) -> None:
    """Test CCRy gate with different decompositions and angles."""
    thetas = np.arange(0, 11) / 10

    for theta in thetas:

        @guppy
        @no_type_check
        def apply_ccry(qs: array[qubit, 3]) -> None:
            ccry_fun(qs[0], qs[1], qs[2], angle(comptime(theta)))  # noqa: B023

        u_guppy = get_unitary(apply_ccry, 3)
        u_exact = ccry_unitary(theta * np.pi)
        assert_allclose_ignorephase(u_guppy, u_exact)
