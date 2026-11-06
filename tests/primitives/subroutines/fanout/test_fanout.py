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
from itertools import product
from typing import no_type_check

import numpy as np
import pytest
from numpy.typing import NDArray
from guppylang import guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.builtins import array, comptime, control, dagger, output
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
from guppyalgos.primitives.subroutines.parity import parity_sequential
from guppyalgos.primitives.utilities.gates import transversal
from guppyalgos.testing import Endianness, assert_allclose_ignorephase, get_unitary
from guppyalgos.utils import bits_to_int, qarray
from guppyalgos.utils.guppy.unsafe_borrow import (
    _unsafe_array_borrow_slice,
    _unsafe_array_unborrow_slice,
)


_ZERO_BITS = [False, False, False, False, False, False]
_ONE_BITS = [True, True, True, True, True, True]
_ARBITRARY_BITS = [False, True, True, False, True, False]
_XOR_BITS = [True, False, False, True, False, True]


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


def _fanout_output(target: bool, bits: list[bool]) -> tuple[bool, list[bool]]:
    return target, [bit ^ target for bit in bits]


def _parity_output(target: bool, bits: list[bool]) -> tuple[bool, list[bool]]:
    return target ^ bits[0] ^ bits[1], bits


def _expected_unitary(
    expected_output: Callable[[bool, list[bool]], tuple[bool, list[bool]]],
    *,
    controlled: bool,
    n_controls: int = 1,
    n_data: int = 2,
) -> NDArray[np.complex128]:
    """Build a permutation matrix from a primitive's output rule.

    Bits are little endian: external controls, target, then data qubits.
    Each column represents an input bitstring; its single entry of 1 selects
    the corresponding output bitstring. All other amplitudes are zero.

    With ``controlled=False``, apply the primitive for every external control
    bitstring. With ``controlled=True``, apply it only when all external
    controls are 1, using the identity otherwise. External controls are always
    preserved. With no external controls, both cases apply the primitive.

    Both primitives are self-adjoint, so forward and dagger share the
    uncontrolled matrix; control and controlled dagger share the controlled
    matrix. This helper describes their phase-free permutations only.
    """
    n_qubits = n_controls + 1 + n_data
    unitary = np.zeros((2**n_qubits, 2**n_qubits), dtype=np.complex128)
    for input_bits in product([False, True], repeat=n_qubits):
        external_bits = list(input_bits[:n_controls])
        input_target = input_bits[n_controls]
        input_data = list(input_bits[n_controls + 1 :])
        if controlled and not all(external_bits):
            output_target, output_data = input_target, input_data
        else:
            output_target, output_data = expected_output(input_target, input_data)

        input_index = bits_to_int(list(input_bits))
        output_index = bits_to_int([*external_bits, output_target, *output_data])
        unitary[output_index, input_index] = 1
    return unitary


@pytest.mark.parametrize(
    ("operation", "expected_output"),
    [
        pytest.param(fanout_basic, _fanout_output, id="fanout-basic"),
        pytest.param(parity_sequential, _parity_output, id="parity-sequential"),
    ],
)
@pytest.mark.parametrize("mode", ["forward", "dagger", "control", "controlled_dagger"])
def test_basic_modifiers(
    operation: GuppyFunctionDefinition[[qubit, array[qubit, 2]], None],
    expected_output: Callable[[bool, list[bool]], tuple[bool, list[bool]]],
    mode: str,
) -> None:
    """Check each primitive and modifier against its full permutation unitary."""

    @guppy
    @no_type_check
    def circuit(qs: array[qubit, 4]) -> None:
        """Adapt the primitive to get_unitary's single borrowed-array argument.

        Temporarily split the array into the external control/target prefix
        and the data register, then restore all qubits to the original array.
        """
        controls, data, suffix = _unsafe_array_borrow_slice(qs, 2, 2, 0)
        if comptime(mode == "forward"):
            operation(controls[1], data)
        elif comptime(mode == "dagger"):
            with dagger:
                operation(controls[1], data)
        elif comptime(mode == "control"):
            with control(controls[0]):
                operation(controls[1], data)
        else:
            with control(controls[0]), dagger:
                operation(controls[1], data)
        _unsafe_array_unborrow_slice(qs, controls, data, suffix)

    expected = _expected_unitary(
        expected_output, controlled=mode in ("control", "controlled_dagger")
    )
    n_extra = int(
        operation is fanout_basic and mode in ("control", "controlled_dagger")
    )
    assert_allclose_ignorephase(
        get_unitary(circuit, 4, endianness=Endianness.LITTLE, n_extra_qubits=n_extra),
        expected,
    )


@pytest.mark.parametrize(
    ("n_controls", "n_data"), [(1, 0), (2, 0), (2, 1), (2, 3), (3, 2)]
)
@pytest.mark.parametrize(
    "daggered", [False, True], ids=["control", "controlled-dagger"]
)
def test_fanout_basic_multiple_controls(
    n_controls: int, n_data: int, daggered: bool
) -> None:
    """Check shared-flag controls and clean work qubits against the full unitary."""
    n_qubits = n_controls + 1 + n_data

    @guppy
    @no_type_check
    def circuit(qs: array[qubit, n_qubits]) -> None:
        controls, targets, suffix = _unsafe_array_borrow_slice(
            qs, n_controls, comptime(1 + n_data), 0
        )
        intrinsic, data, empty = _unsafe_array_borrow_slice(targets, 1, n_data, 0)
        with control(controls):
            if comptime(daggered):
                with dagger:
                    fanout_basic(intrinsic[0], data)
            else:
                fanout_basic(intrinsic[0], data)
        _unsafe_array_unborrow_slice(targets, intrinsic, data, empty)
        _unsafe_array_unborrow_slice(qs, controls, targets, suffix)

    expected = _expected_unitary(
        _fanout_output, controlled=True, n_controls=n_controls, n_data=n_data
    )
    # The predicate includes the intrinsic control: one flag plus CNX ancillas.
    n_extra = 1 + n_controls // 2
    actual = get_unitary(
        circuit, n_qubits, endianness=Endianness.LITTLE, n_extra_qubits=n_extra
    )
    assert_allclose_ignorephase(actual, expected)
