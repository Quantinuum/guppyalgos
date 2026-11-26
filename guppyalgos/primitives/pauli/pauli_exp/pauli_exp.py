"""Pauli Exponential Gadget Implementation."""

from __future__ import annotations
from guppylang import comptime, guppy

from guppylang.std.builtins import Daggerable, Unitary, array, dagger, nat
from guppylang.std.angles import angle
from guppylang.std.quantum import qubit, rz

from guppyalgos.primitives.pauli import pauli_to_z_basis
from guppyalgos.primitives.subroutines.conjugation import ConjugationBoxParams
from guppyalgos.primitives.subroutines.ladders import (
    CXLadderLinear,
    CXLadderLog,
    Ladder,
)

import zixy.qubit.pauli as zqp
from typing import no_type_check


# The built-in ladders implement the Guppy protocol structurally.
type _CXLadder = type[Ladder] | type[CXLadderLinear] | type[CXLadderLog]


def _pauli_parity_basis[n: nat](
    pauli_string: zqp.String, n_qubits: int, cx_ladder: _CXLadder, support: list[int]
) -> Daggerable[[array[qubit, n]], None]:
    """Map a nonidentity Pauli to Z on the last supported qubit."""
    basis_change = pauli_to_z_basis(pauli_string, n_qubits)

    @guppy
    @no_type_check
    def ladder(qs: array[qubit, comptime(n_qubits)], inverse: bool @ comptime) -> None:
        indices = comptime(support)
        subset = array(qs.take(i) for i in indices)
        gates = cx_ladder()
        if inverse:
            gates.ascending_dagger(subset)
        else:
            gates.ascending(subset)
        for i in range(comptime(len(support))):
            qs.put(subset.take(i), indices[i])
        subset.discard_all_taken()

    @guppy.unitary
    class parity_basis:
        @guppy
        @no_type_check
        def __call__(qs: array[qubit, comptime(n_qubits)]) -> None:
            """Change basis and compute parity."""
            basis_change(qs)
            if comptime(len(support) > 1):
                ladder(qs, False)

        @guppy
        @no_type_check
        def daggered(qs: array[qubit, comptime(n_qubits)]) -> None:
            """Uncompute parity and restore the basis."""
            if comptime(len(support) > 1):
                ladder(qs, True)
            with dagger:
                basis_change(qs)

    return parity_basis


def pauli_exp[n_state_q: nat](
    pauli_string: zqp.String,
    n_qubits: int,
    cx_ladder: _CXLadder = CXLadderLog,
    rz_method: Unitary[[qubit, angle], None] = rz,
) -> Unitary[[array[qubit, n_state_q], angle], None]:
    r"""Generate a guppy function to apply the exponential of a Pauli string.

    The Pauli exponential is implemented using the standard decomposition into basis
    changes, CX ladder and RZ rotation. The CX ladder method and RZ decomposition
    method are provided as inputs to allow for flexibility in the implementation.
    The Linear depth CXLadder ladder uses a staircase cascade of CX gates, while the
    Logarithmic depth CXLadder ladder uses a higher CX overhead to achieve
    logarithmic depth.

    $$
    e^{-i \frac{\theta}{2} P} = B^{\dagger} \cdot CX_{ladder}^{\dagger}
    \cdot R_Z(\theta) \cdot CX_{ladder} \cdot B
    $$

    The gadget supports control and dagger and requires a unitary RZ callback.
    External controls apply only to the terminal rotation.

    Where $P$ is the Pauli string to be exponentiated, $B$ is the basis change
    to the Z basis, $CX_{ladder}$ is the CX ladder entangling the qubits acted on
    by the Pauli string, and $R_Z(\theta)$ is the RZ rotation on the target qubit
    of the ladder.

    .. code-block:: python3

        from guppyalgos.primitives.pauli.pauli_exp import pauli_exp
        from guppylang.std.quantum import rz, qubit
        from guppyalgos.primitives.subroutines.ladders import CXLadderLinear
        import zixy.qubit.pauli as zqp
        n_state_qubits = 4
        pauli_string = zqp.String.from_str("Z0 X1 Y2 Z3", n_state_qubits)
        cx_ladder_method = CXLadderLinear
        rz_method = rz
        pauli_gadget = pauli_exp(
            pauli_string, n_state_qubits, cx_ladder_method, rz_method
        )


    Args:
        pauli_string (zqp.String): zixy String representing the Pauli to be
            exponentiated
        n_qubits: number of qubits in the register to apply the Pauli exponential on
        cx_ladder (Ladder): CXLadder method to use for the CX ladder
        rz_method: Unitary RZ implementation.

    """
    if pauli_string.is_identity():
        raise ValueError(
            "Pauli exponential requires at least 1 non-identity Pauli operators"
        )

    support = sorted(pauli_string.get_dict())
    parity_basis = _pauli_parity_basis(pauli_string, n_qubits, cx_ladder, support)
    target = support[-1]

    @guppy(unitary=True)
    @no_type_check
    def rotate(qreg: array[qubit, comptime(n_qubits)], theta: angle) -> None:
        """Rotate the qubit holding the parity."""
        rz_method(qreg[comptime(target)], theta)

    @guppy(unitary=True)
    @no_type_check
    def pauli_gadget_fn(qreg: array[qubit, comptime(n_qubits)], theta: angle) -> None:
        box = ConjugationBoxParams(parity_basis, rotate)
        box.compose(qreg, theta)

    return pauli_gadget_fn
