"""Tests for flat basic and measurement-assisted fanout callables.

The suite checks three uses of fanout with a control prepared in ``|1>``:

* Compute copies the control into a zero-initialized target register.
* Uncompute restores a target register correlated with the control to zero.
* Parity fanout XORs the control into an arbitrary target bitstring.

The basic and parity callables support all three uses. The specialized
measurement compute and uncompute callables are each tested only against
their required target-register input state.
"""

from collections.abc import Callable
from typing import no_type_check

import numpy as np
import pytest
from guppylang import guppy
from guppylang.std.builtins import array, control, dagger, comptime, output
from guppylang.std.quantum import (
    collect_measurements,
    h,
    measure,
    measure_array,
    qubit,
    x,
)

from guppyalgos.primitives.subroutines.fanout import (
    fanout_basic,
    fanout_log,
    fanout_measurement_compute,
    fanout_measurement_compute_total_qubits,
    fanout_measurement_parity,
    fanout_measurement_parity_total_qubits,
    fanout_measurement_uncompute,
)
from guppyalgos.primitives.utilities.gates import transversal
from guppyalgos.utils import qarray
from guppyalgos.testing import assert_allclose_ignorephase, get_unitary
from guppyalgos.utils.guppy.unsafe_borrow import (
    _unsafe_array_borrow_slice,
    _unsafe_array_unborrow_slice,
)

_ZERO_BITS = [False, False, False, False, False, False]
_ONE_BITS = [True, True, True, True, True, True]
_ARBITRARY_BITS = [False, True, True, False, True, False]
_XOR_BITS = [True, False, False, True, False, True]


@pytest.mark.parametrize("mode", ["forward", "dagger", "control", "controlled_dagger"])
def test_fanout_basic_modifiers(mode: str) -> None:
    """Check the self-adjoint operation and its controlled matrix action."""

    @guppy
    @no_type_check
    def circuit(qs: array[qubit, 4]) -> None:
        prefix, data, suffix = _unsafe_array_borrow_slice(qs, 2, 2, 0)
        if comptime(mode == "forward"):
            fanout_basic(prefix[1], data)
        elif comptime(mode == "dagger"):
            with dagger:
                fanout_basic(prefix[1], data)
        elif comptime(mode == "control"):
            with control(prefix[0]):
                fanout_basic(prefix[1], data)
        else:
            with control(prefix[0]), dagger:
                fanout_basic(prefix[1], data)
        _unsafe_array_unborrow_slice(qs, prefix, data, suffix)

    # |control, data0, data1>: control=1 flips both data bits.
    action = np.eye(8, dtype=np.complex128)[:, [0, 1, 2, 3, 7, 6, 5, 4]]
    expected = np.eye(16, dtype=np.complex128)
    expected[8:, 8:] = action
    if mode in ("forward", "dagger"):
        expected[:8, :8] = action
    assert_allclose_ignorephase(get_unitary(circuit, 4), expected)


def _basic_total_qubits(n_target_qubits: int) -> int:
    """Count the control and target qubits."""
    return n_target_qubits + 1


@pytest.mark.parametrize(
    (
        "fanout_operation",
        "control_bit",
        "initial_bits",
        "expected_bits",
        "total_qubits_fn",
    ),
    [
        # Compute: |1>|000000> -> |1>|111111>.
        pytest.param(
            fanout_basic,
            True,
            _ZERO_BITS,
            _ONE_BITS,
            _basic_total_qubits,
            id="basic-compute",
        ),
        pytest.param(
            fanout_log,
            True,
            _ZERO_BITS,
            _ONE_BITS,
            _basic_total_qubits,
            id="log-compute",
        ),
        pytest.param(
            fanout_measurement_compute,
            True,
            _ZERO_BITS,
            _ONE_BITS,
            fanout_measurement_compute_total_qubits,
            id="measurement-compute",
        ),
        pytest.param(
            fanout_measurement_parity,
            True,
            _ZERO_BITS,
            _ONE_BITS,
            fanout_measurement_parity_total_qubits,
            id="measurement-parity-compute",
        ),
        # Uncompute: |1>|111111> -> |1>|000000>.
        pytest.param(
            fanout_basic,
            True,
            _ONE_BITS,
            _ZERO_BITS,
            _basic_total_qubits,
            id="basic-uncompute",
        ),
        pytest.param(
            fanout_log,
            True,
            _ONE_BITS,
            _ZERO_BITS,
            _basic_total_qubits,
            id="log-uncompute",
        ),
        pytest.param(
            fanout_measurement_uncompute,
            True,
            _ONE_BITS,
            _ZERO_BITS,
            _basic_total_qubits,
            id="measurement-uncompute",
        ),
        pytest.param(
            fanout_measurement_parity,
            True,
            _ONE_BITS,
            _ZERO_BITS,
            fanout_measurement_parity_total_qubits,
            id="measurement-parity-uncompute",
        ),
        # General: XOR |1> into an arbitrary target bitstring.
        pytest.param(
            fanout_basic,
            True,
            _ARBITRARY_BITS,
            _XOR_BITS,
            _basic_total_qubits,
            id="basic-general",
        ),
        pytest.param(
            fanout_log,
            True,
            _ARBITRARY_BITS,
            _XOR_BITS,
            _basic_total_qubits,
            id="log-general",
        ),
        pytest.param(
            fanout_measurement_parity,
            True,
            _ARBITRARY_BITS,
            _XOR_BITS,
            fanout_measurement_parity_total_qubits,
            id="measurement-parity",
        ),
        # A zero control leaves arbitrary targets unchanged.
        pytest.param(
            fanout_basic,
            False,
            _ARBITRARY_BITS,
            _ARBITRARY_BITS,
            _basic_total_qubits,
            id="basic-control-zero",
        ),
        pytest.param(
            fanout_log,
            False,
            _ARBITRARY_BITS,
            _ARBITRARY_BITS,
            _basic_total_qubits,
            id="log-control-zero",
        ),
        pytest.param(
            fanout_measurement_parity,
            False,
            _ARBITRARY_BITS,
            _ARBITRARY_BITS,
            fanout_measurement_parity_total_qubits,
            id="measurement-parity-control-zero",
        ),
    ],
)
def test_fanout(
    fanout_operation,
    control_bit: bool,
    initial_bits: list[bool],
    expected_bits: list[bool],
    total_qubits_fn: Callable[[int], int],
) -> None:
    """Apply a flat Guppy fanout callable directly to all target qubits.

    Each parameter row supplies the callable's valid initial target state,
    the expected state after fanout, and the qubit-count function needed by
    the emulator. Measurement-assisted implementations therefore receive
    their ancilla qubits, while basic fanout uses only the control and target
    register.

    The final assertions also verify that fanout preserves the control rather
    than consuming or changing it.
    """
    n_target_qubits = len(initial_bits)

    @guppy
    @no_type_check
    def main() -> None:
        control = qubit()
        target_qreg = qarray(n_target_qubits)
        bits = initial_bits

        if control_bit:
            x(control)
        for bit in range(n_target_qubits):
            if bits[bit]:
                x(target_qreg[bit])

        fanout_operation(control, target_qreg)

        output("control", measure(control).read())
        output("state", collect_measurements(measure_array(target_qreg)))

    result = (
        main.emulator(n_qubits=total_qubits_fn(n_target_qubits))
        .with_seed(42)
        .with_shots(1)
        .run()
        .collated_shots()[0]
    )
    assert result["state"][0] == expected_bits
    assert result["control"][0] == control_bit


@pytest.mark.parametrize(
    ("fanout_operation", "total_qubits_fn"),
    [
        pytest.param(fanout_basic, _basic_total_qubits, id="basic"),
        pytest.param(fanout_log, _basic_total_qubits, id="log"),
        pytest.param(
            fanout_measurement_parity,
            fanout_measurement_parity_total_qubits,
            id="measurement-parity",
        ),
    ],
)
def test_fanout_preserves_all_plus_state(
    fanout_operation,
    total_qubits_fn: Callable[[int], int],
) -> None:
    """Leave an all-plus state unchanged, including all relative phases."""
    n_target_qubits = len(_ZERO_BITS)

    @guppy
    @no_type_check
    def main() -> None:
        control = qubit()
        target_qreg = qarray(n_target_qubits)

        h(control)
        transversal(h, target_qreg)
        fanout_operation(control, target_qreg)
        h(control)
        transversal(h, target_qreg)

        output("control", measure(control).read())
        output("state", collect_measurements(measure_array(target_qreg)))

    result = (
        main.emulator(n_qubits=total_qubits_fn(n_target_qubits))
        .with_seed(42)
        .with_shots(1)
        .run()
        .collated_shots()[0]
    )
    assert result["state"][0] == _ZERO_BITS
    assert result["control"][0] == 0
