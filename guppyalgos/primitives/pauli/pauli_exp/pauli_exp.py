"""Pauli Exponential Gadget Implementation."""

from __future__ import annotations
from guppylang import guppy, comptime

from guppylang.std.builtins import array, control, frozenarray, nat, owned
from guppylang.std.angles import angle
from guppylang.std.quantum import qubit, rz, crz

from guppyalgos.primitives.pauli import pauli_to_z_basis
from guppyalgos.primitives.subroutines.ladders import CXLadderLog, Ladder
from guppyalgos.primitives.measurement.utils import discard_array_zero
from guppyalgos.utils.guppy.unsafe_borrow import (
    _unsafe_array_borrow,
    _unsafe_array_unborrow,
)
from guppyalgos.utils import qarray

import zixy.qubit.pauli as zqp
from typing import no_type_check

from guppylang.defs import GuppyFunctionDefinition


@guppy
@no_type_check
def _take_pauli_qubits[n_qubits: nat, n_paulis: nat](
    qreg: array[qubit, n_qubits],
    indices: frozenarray[int, n_paulis],
) -> tuple[array[qubit, n_paulis], array[qubit, n_qubits]]:
    borrowed_qreg = _unsafe_array_borrow(qreg)
    qubit_subset = array(borrowed_qreg.take(indices[i]) for i in range(n_paulis))
    return qubit_subset, borrowed_qreg


@guppy
@no_type_check
def _return_pauli_qubits[n_qubits: nat, n_paulis: nat](
    qreg: array[qubit, n_qubits],
    qubit_subset: array[qubit, n_paulis] @ owned,  # ty: ignore[not-subscriptable]
    borrowed_qreg: array[qubit, n_qubits] @ owned,  # ty: ignore[not-subscriptable]
    indices: frozenarray[int, n_paulis],
) -> None:
    for i in range(n_paulis):
        borrowed_qreg.put(qubit_subset.take(i), indices[i])
    qubit_subset.discard_all_taken()
    _unsafe_array_unborrow(qreg, borrowed_qreg)


def pauli_exp[n_state_q: nat](
    pauli_string: zqp.String,
    n_qubits: int,
    cx_ladder: type[Ladder] = CXLadderLog,
    rz_method: GuppyFunctionDefinition[[qubit, angle], None] = rz,
) -> GuppyFunctionDefinition[[array[qubit, n_state_q], angle], None]:
    r"""Generate a guppy function to apply the exponential of a Pauli string.

    The Pauli exponential is implemented using the standard decomposition into basis
    changes, CX ladder and RZ rotation. The CX ladder method and RZ decomposition
    method are provided as inputs to allow for flexibility in the implementation.
    The Linear depth CXLadder ladder uses a staircase cascade of CX gates, while the
    Logarithmic depth CXLadder ladder uses a higher CX overhead to achieve
    logarithmic depth.

    $$
    e^{-i \frac{\\theta}{2} P} = B^{\dagger} \cdot CX_{ladder}^{\dagger}
    \cdot R_Z(\theta) \cdot CX_{ladder} \cdot B
    $$

    Where $P$ is the Pauli string to be exponentiated, $B$ is the basis change
    to the Z basis, $CX_{ladder}$ is the CX ladder entangling the qubits acted on
    by the Pauli string, and $R_Z(\theta)$ is the RZ rotation on the target qubit
    of the ladder.

    .. code-block:: python3

        from guppyalgos.primitives.pauli.pauli_exp import pauli_exp
        from guppylang.std.quantum import rz, qubit
        from guppyalgos.primitives.subroutines.ladders import LinearCXLadder
        import zixy.qubit.pauli as zqp
        n_state_qubits = 4
        pauli_string = zqp.String.from_str("Z0 X1 Y2 Z3", n_state_qubits)
        cx_ladder_method = LinearCXLadder
        rz_method = rz
        pauli_gadget = pauli_exp(
            pauli_string, n_state_qubits, cx_ladder_method, rz_method
        )


    Args:
        pauli_string (zqp.String): zixy String representing the Pauli to be
            exponentiated
        n_qubits: number of qubits in the register to apply the Pauli exponential on
        cx_ladder (Ladder): CXLadder method to use for the CX ladder
        rz_method (GuppyFunctionDefinition): RZ decomposition method to use for the RZ

    """
    rz_flags = getattr(rz_method.wrapped, "unitary_flags", None)
    has_unitary_rz = getattr(rz_flags, "name", None) == "Unitary"

    if pauli_string.is_identity():
        if not has_unitary_rz:

            @guppy
            @no_type_check
            def identity_gadget(qreg: array[qubit, n_qubits], angle: angle) -> None:
                pass

            return identity_gadget

        @guppy.unitary
        class identity_gadget:
            @guppy
            @no_type_check
            def __call__(qreg: array[qubit, n_qubits], angle: angle) -> None:
                pass

            @guppy
            @no_type_check
            def controlled[n_controls: nat](
                qreg: array[qubit, n_qubits],
                angle: angle,
                controls: array[qubit, n_controls],
            ) -> None:
                if n_controls == 1:
                    rz_method(controls[0], -angle / 2)
                else:
                    phase_qreg = qarray(1)
                    with control(controls):
                        rz_method(phase_qreg[0], angle)
                    discard_array_zero(phase_qreg)

            @guppy
            @no_type_check
            def daggered(qreg: array[qubit, n_qubits], angle: angle) -> None:
                pass

            @guppy
            @no_type_check
            def ctrl_daggered[n_controls: nat](
                qreg: array[qubit, n_qubits],
                angle: angle,
                controls: array[qubit, n_controls],
            ) -> None:
                if n_controls == 1:
                    rz_method(controls[0], angle / 2)
                else:
                    phase_qreg = qarray(1)
                    with control(controls):
                        rz_method(phase_qreg[0], -angle)
                    discard_array_zero(phase_qreg)

        return identity_gadget

    basis_change = pauli_to_z_basis(pauli_string=pauli_string, size=n_qubits)
    basis_change_dagger = pauli_to_z_basis(
        pauli_string=pauli_string, size=n_qubits, dagger=True
    )

    pauli_indices = list(pauli_string.get_dict().keys())

    if not has_unitary_rz:

        @guppy
        @no_type_check
        def pauli_gadget_fn(qreg: array[qubit, n_qubits], angle: angle) -> None:
            ladder = cx_ladder()
            basis_change(qreg)
            qubit_subset, borrowed_qreg = _take_pauli_qubits(
                qreg, comptime(pauli_indices)
            )
            ladder.ascending(qubit_subset)
            rz_method(qubit_subset[len(qubit_subset) - 1], angle)
            ladder.ascending_dagger(qubit_subset)
            _return_pauli_qubits(
                qreg, qubit_subset, borrowed_qreg, comptime(pauli_indices)
            )
            basis_change_dagger(qreg)

        return pauli_gadget_fn

    @guppy.unitary
    class pauli_gadget_fn:
        @guppy
        @no_type_check
        def __call__(qreg: array[qubit, n_qubits], angle: angle) -> None:
            ladder = cx_ladder()
            basis_change(qreg)
            qubit_subset, borrowed_qreg = _take_pauli_qubits(
                qreg, comptime(pauli_indices)
            )
            ladder.ascending(qubit_subset)
            rz_method(qubit_subset[len(qubit_subset) - 1], angle)
            ladder.ascending_dagger(qubit_subset)
            _return_pauli_qubits(
                qreg, qubit_subset, borrowed_qreg, comptime(pauli_indices)
            )
            basis_change_dagger(qreg)

        @guppy
        @no_type_check
        def controlled[n_controls: nat](
            qreg: array[qubit, n_qubits],
            angle: angle,
            controls: array[qubit, n_controls],
        ) -> None:
            ladder = cx_ladder()
            basis_change(qreg)
            qubit_subset, borrowed_qreg = _take_pauli_qubits(
                qreg, comptime(pauli_indices)
            )
            ladder.ascending(qubit_subset)
            with control(controls):
                rz_method(qubit_subset[len(qubit_subset) - 1], angle)
            ladder.ascending_dagger(qubit_subset)
            _return_pauli_qubits(
                qreg, qubit_subset, borrowed_qreg, comptime(pauli_indices)
            )
            basis_change_dagger(qreg)

        @guppy
        @no_type_check
        def daggered(qreg: array[qubit, n_qubits], angle: angle) -> None:
            ladder = cx_ladder()
            basis_change(qreg)
            qubit_subset, borrowed_qreg = _take_pauli_qubits(
                qreg, comptime(pauli_indices)
            )
            ladder.ascending(qubit_subset)
            rz_method(qubit_subset[len(qubit_subset) - 1], -angle)
            ladder.ascending_dagger(qubit_subset)
            _return_pauli_qubits(
                qreg, qubit_subset, borrowed_qreg, comptime(pauli_indices)
            )
            basis_change_dagger(qreg)

        @guppy
        @no_type_check
        def ctrl_daggered[n_controls: nat](
            qreg: array[qubit, n_qubits],
            angle: angle,
            controls: array[qubit, n_controls],
        ) -> None:
            ladder = cx_ladder()
            basis_change(qreg)
            qubit_subset, borrowed_qreg = _take_pauli_qubits(
                qreg, comptime(pauli_indices)
            )
            ladder.ascending(qubit_subset)
            with control(controls):
                rz_method(qubit_subset[len(qubit_subset) - 1], -angle)
            ladder.ascending_dagger(qubit_subset)
            _return_pauli_qubits(
                qreg, qubit_subset, borrowed_qreg, comptime(pauli_indices)
            )
            basis_change_dagger(qreg)

    return pauli_gadget_fn


# Legacy implementation, to be deleted once everything uses the custom modifier
def cntrl_pauli_exp[n_state_q: nat](
    pauli_string: zqp.String,
    n_qubits: int,
    cx_ladder: type[Ladder] = CXLadderLog,
    controlled_rz_method: GuppyFunctionDefinition[[qubit, qubit, angle], None] = crz,
    rz_method: GuppyFunctionDefinition[[qubit, angle], None] = rz,
) -> GuppyFunctionDefinition[[qubit, array[qubit, n_state_q], angle], None]:
    r"""Generate a controlled Pauli exponential using the conjugation pattern.

    The controlled Pauli exponential follows the same conjugation pattern as
    :func:`pauli_exp`: conjugate the active support into the Z basis, accumulate the
    parity onto one target qubit with a CX ladder, apply a terminal controlled RZ
    rotation, and then uncompute the ladder and basis change. The CX ladder method
    and controlled RZ decomposition method are provided as inputs to allow for
    flexibility in the implementation.

    $$
    C\left(e^{-i \frac{\theta}{2} P}\right) = B^{\dagger} \cdot
    CX_{ladder}^{\dagger} \cdot CR_Z(\theta) \cdot CX_{ladder} \cdot B
    $$

    Where $P$ is the Pauli string to be exponentiated, $B$ is the basis change
    to the Z basis, $CX_{ladder}$ is the CX ladder entangling the qubits acted on
    by the Pauli string, and $CR_Z(\theta)$ is the controlled RZ rotation on the
    target qubit of the ladder. For the all-identity string, this reduces to a
    control-only phase, which is currently implemented via ``rz_method`` until a
    dedicated phase primitive is available.

    Args:
        pauli_string (zqp.String): zixy String representing the Pauli to be
            exponentiated
        n_qubits: number of qubits in the register to apply the Pauli exponential on
        cx_ladder (Ladder): CX ladder, implements Ladder protocol.
        controlled_rz_method (GuppyFunctionDefinition): Controlled RZ decomposition
            method to use for the terminal controlled rotation
        rz_method (GuppyFunctionDefinition): Single-qubit phase implementation used
            for the all-identity controlled exponential

    """
    if pauli_string.is_identity():

        @guppy.comptime
        @no_type_check
        def cntrl_pauli_gadget_fn(
            control: qubit,
            qreg: array[qubit, n_qubits],
            theta: angle,
        ) -> None:
            rz_method(control, -theta / 2)

        return cntrl_pauli_gadget_fn

    basis_change = pauli_to_z_basis(pauli_string=pauli_string, size=n_qubits)

    basis_change_dagger = pauli_to_z_basis(
        pauli_string=pauli_string, size=n_qubits, dagger=True
    )

    pauli_indices = tuple(pauli_string.get_dict().keys())

    @guppy.comptime
    @no_type_check
    def cntrl_pauli_gadget_fn(
        control: qubit,
        qreg: array[qubit, n_qubits],
        theta: angle,
    ) -> None:
        ladder = cx_ladder()
        qubit_subset = [qreg[i] for i in pauli_indices]

        basis_change(qreg)

        if len(qubit_subset) > 1:
            ladder.ascending(qubit_subset)

        controlled_rz_method(control, qubit_subset[-1], theta)

        if len(qubit_subset) > 1:
            ladder.ascending_dagger(qubit_subset)

        basis_change_dagger(qreg)

    return cntrl_pauli_gadget_fn
