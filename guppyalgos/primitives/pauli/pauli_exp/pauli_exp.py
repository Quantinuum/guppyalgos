"""Pauli Exponential Gadget Implementation."""

from __future__ import annotations
from guppylang import guppy

from guppylang.std.builtins import array, nat
from guppylang.std.angles import angle
from guppylang.std.quantum import qubit, rz, crz

from guppyalgos.primitives.pauli import pauli_to_z_basis
from guppyalgos.primitives.pauli.pauli import _daggered_basis_change
from guppyalgos.primitives.subroutines.ladders import CXLadderLog, Ladder

import zixy.qubit.pauli as zqp
from typing import no_type_check

from guppylang.defs import GuppyFunctionDefinition


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
    if pauli_string.is_identity():
        raise ValueError(
            "Pauli exponential requires at least 1 non-identity Pauli operators"
        )

    basis_change = pauli_to_z_basis(pauli_string=pauli_string, size=n_qubits)
    restore_basis = _daggered_basis_change(basis_change, n_qubits)

    pauli_indices = tuple(pauli_string.get_dict().keys())

    @guppy.comptime
    @no_type_check
    def pauli_gadget_fn(qreg: array[qubit, n_qubits], angle: angle) -> None:
        ladder = cx_ladder()
        qubit_subset = [qreg[i] for i in pauli_indices]

        basis_change(qreg)

        if len(qubit_subset) > 1:
            ladder.ascending(qubit_subset)

        rz_method(qubit_subset[-1], angle)

        if len(qubit_subset) > 1:
            ladder.ascending_dagger(qubit_subset)

        restore_basis(qreg)

    return pauli_gadget_fn


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
    restore_basis = _daggered_basis_change(basis_change, n_qubits)

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

        restore_basis(qreg)

    return cntrl_pauli_gadget_fn
