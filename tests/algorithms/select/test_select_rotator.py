"""Compilation and numerical tests for rotated selection circuits."""

from math import cos, pi, sin
from typing import no_type_check

import numpy as np
import pytest
from guppylang import comptime, guppy
from guppylang.std.builtins import Function, array, nat, owned
from guppylang.std.debug import state_result
from guppylang.std.quantum import cz, discard, discard_array, qubit, x
from selene_sim import Quest

from guppyalgos.primitives.gate_decompositions.cnx.ccu import ccz
from guppyalgos.algorithms.select.qrom import qrom_unary_iteration
from guppyalgos.algorithms.select.qrom.controlled_qrom_unary_iteration import (
    cntrl_qrom_unary_iteration,
)
from guppyalgos.primitives.rotations import (
    GivensCascadePhaseGradient,
    QROMRotations,
    qrom_identity,
)
from guppyalgos.algorithms.select import SelectRotator
from guppyalgos.primitives.state_preparation.phase_gradient import (
    Convention,
    phase_gradient,
)
from guppyalgos.utils import (
    apply_bitstring,
    bits_to_int,
    discard_nested_arrays,
    int_to_bits,
    qarray,
)
from guppyalgos.testing import (
    assert_allclose_ignorephase,
    get_total_state_on_only_specified_registers,
)


_ANGLE_DATA = [
    [[False, False, False], [False, False, False]],
    [[True, False, False], [False, True, False]],
    [[False, True, False], [True, False, False]],
    [[True, True, False], [True, True, False]],
]
_SELECTED_INDEX = 1
_N_DATA_QUBITS = 3
_N_GIVENS = 2
_N_MODES = 3

_qrom_compute = qrom_unary_iteration(_ANGLE_DATA)
_qrom_uncompute = qrom_unary_iteration(_ANGLE_DATA)
_cntrl_qrom = cntrl_qrom_unary_iteration(_ANGLE_DATA)
_prepare_phase_gradient = phase_gradient(
    _N_DATA_QUBITS,
    convention=Convention.Standard,
)


@guppy
@no_type_check
def _cntrl_z[n_rotation_q: nat](
    control_regs: array[qubit, 1],
    rotation_regs: array[qubit, n_rotation_q],
) -> None:
    """Apply controlled Z to the first rotation target."""
    cz(control_regs[0], rotation_regs[0])


@guppy
@no_type_check
def _doubly_cntrl_z[n_rotation_q: nat](
    control_regs: array[qubit, 2],
    rotation_regs: array[qubit, n_rotation_q],
) -> None:
    """Apply doubly controlled Z to the first rotation target."""
    ccz(control_regs[0], control_regs[1], rotation_regs[0])


@guppy
@no_type_check
def _qrom_ccz_subselect[
    n_index_q: nat,
    n_data_q: nat,
    n_givens: nat,
    n_modes: nat,
](
    qrom_compute: Function[
        [
            array[qubit, n_index_q],
            array[array[qubit, n_data_q], n_givens],  # ty: ignore[not-subscriptable]
        ],
        None,
    ],
    qrom_uncompute: Function[
        [
            array[qubit, n_index_q],
            array[array[qubit, n_data_q], n_givens],  # ty: ignore[not-subscriptable]
        ],
        None,
    ],
    index_qreg: array[qubit, n_index_q],
    control_regs: array[qubit, 2],
    rotation_regs: array[qubit, n_modes],
    pg_state: array[qubit, n_data_q] @ owned,  # ty: ignore[not-subscriptable]
) -> None:
    """Build the concrete QROM and doubly-controlled-Z selection under test."""
    data_qregs = array(qarray(n_data_q) for _ in range(n_givens))
    qrom_rotations = QROMRotations(
        qrom_compute,
        GivensCascadePhaseGradient[n_data_q, n_givens, n_modes](pg_state),
        qrom_uncompute,
    )
    select_rotator = SelectRotator(
        qrom_rotations,
        _doubly_cntrl_z[n_modes],
    )
    select_rotator.compose(index_qreg, data_qregs, control_regs, rotation_regs)
    discard_nested_arrays(data_qregs)
    discard_array(select_rotator.qrom_rotations.rotation_box.phase_gradient)


@guppy
@no_type_check
def _cntrl_qrom_cz_subselect[
    n_index_q: nat,
    n_data_q: nat,
    n_givens: nat,
    n_modes: nat,
](
    qrom_compute: Function[
        [
            tuple[qubit, array[qubit, n_index_q]],
            array[array[qubit, n_data_q], n_givens],  # ty: ignore[not-subscriptable]
        ],
        None,
    ],
    qrom_uncompute: Function[
        [
            tuple[qubit, array[qubit, n_index_q]],
            array[array[qubit, n_data_q], n_givens],  # ty: ignore[not-subscriptable]
        ],
        None,
    ],
    index_regs: tuple[qubit, array[qubit, n_index_q]],
    control_regs: array[qubit, 1],
    rotation_regs: array[qubit, n_modes],
    pg_state: array[qubit, n_data_q] @ owned,  # ty: ignore[not-subscriptable]
) -> None:
    """Build the concrete controlled-QROM and controlled-Z selection under test."""
    data_qregs = array(qarray(n_data_q) for _ in range(n_givens))
    qrom_rotations = QROMRotations(
        qrom_compute,
        GivensCascadePhaseGradient[n_data_q, n_givens, n_modes](pg_state),
        qrom_uncompute,
    )
    select_rotator = SelectRotator(
        qrom_rotations,
        _cntrl_z[n_modes],
    )
    select_rotator.compose(index_regs, data_qregs, control_regs, rotation_regs)
    discard_nested_arrays(data_qregs)
    discard_array(select_rotator.qrom_rotations.rotation_box.phase_gradient)


@guppy
@no_type_check
def _bundled_cntrl_qrom(
    index_regs: tuple[qubit, array[qubit, 2]],
    data_qregs: array[array[qubit, 3], 2],
) -> None:
    """Adapt the concrete controlled QROM to the bundled index-register shape."""
    _cntrl_qrom[array[array[qubit, 3], 2]](
        index_regs[0],
        index_regs[1],
        data_qregs,
    )


def _givens_matrix(pair_index: int, theta: float) -> np.ndarray:
    """Return a neighboring Givens rotation on the three-mode register."""
    matrix = np.eye(2**_N_MODES, dtype=np.complex128)
    c = cos(pi * theta)
    s = sin(pi * theta)

    for basis in range(2**_N_MODES):
        first_position = pair_index
        second_position = pair_index + 1
        basis_bits = int_to_bits(basis, _N_MODES)
        if not basis_bits[first_position] or basis_bits[second_position]:
            continue

        swapped_bits = basis_bits.copy()
        swapped_bits[first_position] = False
        swapped_bits[second_position] = True
        swapped_basis = bits_to_int(swapped_bits)
        matrix[basis, basis] = c
        matrix[basis, swapped_basis] = -s
        matrix[swapped_basis, basis] = s
        matrix[swapped_basis, swapped_basis] = c

    return matrix


def _expected_select_rotator_unitary() -> np.ndarray:
    """Return the paper's ``U_mu^dagger Z_0 U_mu`` operator for the test data."""
    selected_angles = _ANGLE_DATA[_SELECTED_INDEX]
    givens = np.eye(2**_N_MODES, dtype=np.complex128)
    for pair_index, bits in enumerate(selected_angles):
        encoded_integer = bits_to_int(bits)
        theta = float(2.0 * encoded_integer / (2 ** len(bits)))
        givens = _givens_matrix(pair_index, theta) @ givens

    first_mode_mask = 1
    z_first_mode = np.diag(
        [-1.0 if basis & first_mode_mask else 1.0 for basis in range(2**_N_MODES)]
    )

    # Matrix products act right-to-left: rotate into the selected orbital basis
    # with U_mu, apply Z to its first mode, then undo it with U_mu^dagger.
    return givens.conj().T @ z_first_mode @ givens


def _run_select_rotator_on_all_rotation_inputs(
    cntrl_qrom: bool,
    identity_uncompute: bool,
) -> list[np.ndarray]:
    """Run one QROM arrangement on every rotation-register basis state."""

    @guppy
    @no_type_check
    def main(bits: array[bool, 3]) -> None:
        rotation_regs = qarray(3)
        pg_state = qarray(3)
        _prepare_phase_gradient(pg_state)
        apply_bitstring(rotation_regs, bits)

        if comptime(cntrl_qrom):
            index_regs = (qubit(), qarray(2))
            control_regs = qarray(1)
            x(index_regs[0])
            x(index_regs[1][0])
            x(control_regs[0])
            if comptime(identity_uncompute):
                _cntrl_qrom_cz_subselect(
                    _bundled_cntrl_qrom,
                    qrom_identity[
                        tuple[qubit, array[qubit, 2]],
                        array[array[qubit, 3], 2],
                    ],
                    index_regs,
                    control_regs,
                    rotation_regs,
                    pg_state,
                )
            else:
                _cntrl_qrom_cz_subselect(
                    _bundled_cntrl_qrom,
                    _bundled_cntrl_qrom,
                    index_regs,
                    control_regs,
                    rotation_regs,
                    pg_state,
                )
            qrom_control, index_qreg = index_regs
            discard(qrom_control)
            discard_array(index_qreg)
            discard_array(control_regs)
        else:
            index_qreg = qarray(2)
            control_regs = qarray(2)
            x(index_qreg[0])
            x(control_regs[0])
            x(control_regs[1])
            if comptime(identity_uncompute):
                _qrom_ccz_subselect(
                    _qrom_compute[array[array[qubit, 3], 2]],
                    qrom_identity[
                        array[qubit, 2],
                        array[array[qubit, 3], 2],
                    ],
                    index_qreg,
                    control_regs,
                    rotation_regs,
                    pg_state,
                )
            else:
                _qrom_ccz_subselect(
                    _qrom_compute[array[array[qubit, 3], 2]],
                    _qrom_uncompute[array[array[qubit, 3], 2]],
                    index_qreg,
                    control_regs,
                    rotation_regs,
                    pg_state,
                )
            discard_array(index_qreg)
            discard_array(control_regs)

        state_result("result_state", rotation_regs)
        discard_array(rotation_regs)

    emulator = main.emulator(n_qubits=19).with_seed(42)
    output_states = []
    for basis in range(2**_N_MODES):
        result = emulator.run(bits=int_to_bits(basis, _N_MODES))
        states = Quest.extract_states_dict(result.results[0].entries)
        rotation_state, _ = get_total_state_on_only_specified_registers(
            states,
            ["result_state"],
        )
        output_states.append(rotation_state.state)
    return output_states


@pytest.mark.parametrize(
    "cntrl_qrom",
    [
        pytest.param(True, id="cntrl_qrom_cz"),
        pytest.param(False, id="qrom_ccz"),
    ],
)
@pytest.mark.parametrize(
    "identity_uncompute",
    [
        pytest.param(False, id="qrom_uncompute"),
        pytest.param(True, id="identity_uncompute"),
    ],
)
def test_select_rotator_matches_lcu_operator(
    cntrl_qrom: bool,
    identity_uncompute: bool,
) -> None:
    """Check the rotated SELECT unitary against Eq. (31) of arXiv:2501.06165v1.

    With all Pauli and QROM controls enabled, both supported circuit shapes must
    implement ``U_mu^dagger Z_0 U_mu`` on the three rotation modes. ``U_mu`` is
    the two-step Givens cascade selected by the fixed QROM index. The comparison
    covers all eight computational-basis inputs of the rotation register, which
    checks every column of the expected operator.

    The QROM target is restored both with the conventional compute/uncompute pair
    and with identity in the uncompute slot. In the latter arrangement, the
    compute at the end of :meth:`QROMRotations.daggered` clears the data loaded by
    the initial compute.

    """
    expected_operator = _expected_select_rotator_unitary()
    actual_outputs = _run_select_rotator_on_all_rotation_inputs(
        cntrl_qrom,
        identity_uncompute,
    )

    for basis, actual in enumerate(actual_outputs):
        input_state = np.zeros(2**_N_MODES, dtype=np.complex128)
        input_state[basis] = 1.0
        expected = expected_operator @ input_state
        assert_allclose_ignorephase(actual, expected)
