"""Pauli utils for interfacing with zixy."""

from typing import no_type_check

import zixy.qubit.pauli as zqp
from guppylang import comptime, guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.angles import angle
from guppylang.std.builtins import Unitary, array, control, dagger, frozenarray, nat
from guppylang.std.quantum import cx, cy, cz, h, qubit, sdg, x, y, z

from guppyalgos.primitives.utilities.gates import apply_phase


def pauli_to_gate[n_q: nat](
    pauli_string: zqp.String, size: int
) -> GuppyFunctionDefinition[array[qubit, n_q], None]:
    """Produce a guppy function to apply a zixy String.

    Args:
        pauli_string (zqp.String): zixy string for the Pauli to be applied
        size (int): size of the register to apply the Pauli on

    Returns:
        GuppyFunctionDefinition that takes in a qarray[size] and applies the pauli.

    """

    @guppy.comptime
    @no_type_check
    def pauli_gate(q: array[qubit, size]) -> None:
        for idx, p in pauli_string.get_dict().items():
            if idx >= size:
                raise ValueError("Pauli index out of register index range")
            match p:
                case zqp.X:
                    x(q[idx])
                case zqp.Y:
                    y(q[idx])
                case zqp.Z:
                    z(q[idx])

    return pauli_gate


def pauli_to_cntrl_gate[n_q: nat](
    pauli_string: zqp.String, size: int, phase: float = 0.0
) -> GuppyFunctionDefinition[[qubit, array[qubit, n_q]], None]:
    r"""Produce a Guppy function for :math:`e^{i\pi\phi}P` under one control.

    The phase is applied once to the active control branch. If this operation is
    itself controlled, the phase gate must be promoted to ``cphase`` so that it
    is applied only when both controls are active.

    Args:
        pauli_string: Zixy string for the controlled Pauli to be applied.
        size: Size of the register to apply the Pauli on.
        phase: Phase applied to the active control branch, in half-turns.

    Returns:
        GuppyFunctionDefinition that takes in a control qubit and  array[size] and
        applies the pauli controlled on the control qubit.

    """

    @guppy.comptime
    @no_type_check
    def cntrl_pauli_gate(c: qubit, q: array[qubit, size]) -> None:
        for idx, p in pauli_string.get_dict().items():
            if idx >= size:
                raise ValueError("Pauli index out of register index range")
            match p:
                case zqp.X:
                    cx(c, q[idx])
                case zqp.Y:
                    cy(c, q[idx])
                case zqp.Z:
                    cz(c, q[idx])
        apply_phase(c, angle(phase))

    return cntrl_pauli_gate


@guppy(controllable=True)
@no_type_check
def _apply_z_basis[n: nat, nx: nat, ny: nat](
    q: array[qubit, n],
    x_indices: frozenarray[int, nx],
    y_indices: frozenarray[int, ny],
    inverse: bool @ comptime,
) -> None:
    """Apply independent single-qubit basis changes in either direction."""
    for idx in x_indices:
        h(q[idx])
    for idx in y_indices:
        if inverse:
            with dagger:
                sdg(q[idx])
                h(q[idx])
        else:
            sdg(q[idx])
            h(q[idx])


@guppy.unitary
class _z_basis_change[n: nat, nx: nat, ny: nat]:
    """Apply a Pauli basis change with explicit modifiers for the index loops."""

    @guppy
    @no_type_check
    def __call__(
        q: array[qubit, n],
        x_indices: frozenarray[int, nx],
        y_indices: frozenarray[int, ny],
    ) -> None:
        """Map the Pauli string to the Z basis."""
        _apply_z_basis(q, x_indices, y_indices, False)

    @guppy
    @no_type_check
    def daggered(
        q: array[qubit, n],
        x_indices: frozenarray[int, nx],
        y_indices: frozenarray[int, ny],
    ) -> None:
        """Restore the original basis."""
        _apply_z_basis(q, x_indices, y_indices, True)

    @guppy
    @no_type_check
    def controlled[k: nat](
        q: array[qubit, n],
        x_indices: frozenarray[int, nx],
        y_indices: frozenarray[int, ny],
        controls: array[qubit, k],
    ) -> None:
        """Change basis under external controls."""
        with control(controls):
            _apply_z_basis(q, x_indices, y_indices, False)

    @guppy
    @no_type_check
    def ctrl_daggered[k: nat](
        q: array[qubit, n],
        x_indices: frozenarray[int, nx],
        y_indices: frozenarray[int, ny],
        controls: array[qubit, k],
    ) -> None:
        """Restore the basis under external controls."""
        with control(controls):
            _apply_z_basis(q, x_indices, y_indices, True)


def pauli_to_z_basis[n_q: nat](
    pauli_string: zqp.String, size: int
) -> Unitary[[array[qubit, n_q]], None]:
    """Produce a guppy function to change basis to/from Z-basis for a zixy String.

    Maps the given Pauli to the Z-basis on a register of given size. Pair the
    basis change with its inverse to conjugate a Pauli operation.

    Args:
        pauli_string (zqp.String): zixy string for the Pauli to change basis for
        size (int): size of the register to apply the basis change on

    Returns:
        A unitary that applies the basis change to an array of ``size`` qubits.

    """
    paulis = pauli_string.get_dict()
    if any(idx >= size for idx in paulis):
        raise ValueError("Pauli index out of register index range")
    x_indices = [idx for idx, p in paulis.items() if p == zqp.X]
    y_indices = [idx for idx, p in paulis.items() if p == zqp.Y]

    @guppy(unitary=True)
    @no_type_check
    def pauli_gate(q: array[qubit, comptime(size)]) -> None:
        """Apply the basis change for the supplied Pauli string."""
        _z_basis_change(q, comptime(x_indices), comptime(y_indices))

    return pauli_gate
