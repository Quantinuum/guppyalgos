"""Check conjugation boxes using the shared unitary modifier assertions."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import no_type_check

import numpy as np
import pytest
import zixy.qubit.pauli as zqp
from guppylang import guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.angles import angle
from guppylang.std.builtins import array
from guppylang.std.quantum import cx, cz, h, qubit, rx, ry, rz, sdg, x
from numpy.typing import NDArray
from scipy.linalg import expm

from guppyalgos.primitives.subroutines.conjugation import ConjugationBox
from guppyalgos.testing import assert_unitary_modifiers


@dataclass(frozen=True)
class ConjugationCase:
    """Component functions, register width, and expected conjugation matrix."""

    g: GuppyFunctionDefinition
    u: GuppyFunctionDefinition
    width: int
    matrix: NDArray[np.complex128]


def _hamiltonian_unitary(
    hamiltonian: zqp.RealTermSum, theta: float
) -> NDArray[np.complex128]:
    """Exponentiate a Zixy Hamiltonian in the modifier helper's big-endian order."""
    matrix = hamiltonian.to_sparse_matrix(big_endian=True).toarray()
    return expm(-0.5j * np.pi * theta * matrix)


def _basis_rotation_case(theta: float) -> ConjugationCase:
    """Conjugate Rz(theta) by Rx(theta), with theta in half-turns."""

    @guppy(daggerable=True)
    @no_type_check
    def g(qs: array[qubit, 1]) -> None:
        rx(qs[0], angle(theta))

    @guppy(unitary=True)
    @no_type_check
    def u(qs: array[qubit, 1]) -> None:
        rz(qs[0], angle(theta))

    hamiltonian = zqp.RealTermSum.from_str(
        f"({np.cos(np.pi * theta)}, Z0), ({np.sin(np.pi * theta)}, Y0)", 1
    )
    return ConjugationCase(
        g=g,
        u=u,
        width=1,
        matrix=_hamiltonian_unitary(hamiltonian, theta),
    )


def _parallel_rotations_case(theta: float) -> ConjugationCase:
    """Apply matching X rotations using a Hadamard basis change."""

    @guppy(daggerable=True)
    @no_type_check
    def g(qs: array[qubit, 2]) -> None:
        h(qs[0])
        h(qs[1])

    @guppy(unitary=True)
    @no_type_check
    def u(qs: array[qubit, 2]) -> None:
        rz(qs[0], angle(theta))
        rz(qs[1], angle(theta))

    hamiltonian = zqp.RealTermSum.from_str("(1.0, X0), (1.0, X1)", 2)
    return ConjugationCase(
        g=g,
        u=u,
        width=2,
        matrix=_hamiltonian_unitary(hamiltonian, theta),
    )


def _pauli_gadget_case(theta: float) -> ConjugationCase:
    """Apply a ZZZ Pauli gadget using a parity basis change."""

    @guppy(daggerable=True)
    @no_type_check
    def g(qs: array[qubit, 3]) -> None:
        cx(qs[1], qs[0])
        cx(qs[2], qs[0])

    @guppy(unitary=True)
    @no_type_check
    def u(qs: array[qubit, 3]) -> None:
        rz(qs[0], angle(theta))

    hamiltonian = zqp.RealTermSum.from_str("(1.0, Z0 Z1 Z2)", 3)
    return ConjugationCase(
        g=g,
        u=u,
        width=3,
        matrix=_hamiltonian_unitary(hamiltonian, theta),
    )


def _zx_pauli_gadget_case(theta: float) -> ConjugationCase:
    """Apply a ZX Pauli gadget with different operators on the two qubits."""

    @guppy(daggerable=True)
    @no_type_check
    def g(qs: array[qubit, 2]) -> None:
        h(qs[1])
        cx(qs[1], qs[0])

    @guppy(unitary=True)
    @no_type_check
    def u(qs: array[qubit, 2]) -> None:
        rz(qs[0], angle(theta))

    hamiltonian = zqp.RealTermSum.from_str("(1.0, Z0 X1)", 2)
    return ConjugationCase(
        g=g,
        u=u,
        width=2,
        matrix=_hamiltonian_unitary(hamiltonian, theta),
    )


def _xyz_pauli_gadget_case(theta: float) -> ConjugationCase:
    """Apply an XYZ Pauli gadget with an S-dagger/H basis change for Y."""

    @guppy(daggerable=True)
    @no_type_check
    def g(qs: array[qubit, 3]) -> None:
        h(qs[0])
        sdg(qs[1])
        h(qs[1])
        cx(qs[0], qs[2])
        cx(qs[1], qs[2])

    @guppy(unitary=True)
    @no_type_check
    def u(qs: array[qubit, 3]) -> None:
        rz(qs[2], angle(theta))

    hamiltonian = zqp.RealTermSum.from_str("(1.0, X0 Y1 Z2)", 3)
    return ConjugationCase(
        g=g,
        u=u,
        width=3,
        matrix=_hamiltonian_unitary(hamiltonian, theta),
    )


def _two_term_lcu_case(theta: float) -> ConjugationCase:
    """Block encode cos²(pi theta/2) X + sin²(pi theta/2) Z with Ry PREPARE.

    The first qubit selects X on zero and Z on one. The full unitary is
    PREPARE† SELECT PREPARE, retaining the index qubit for modifier checks.
    """

    @guppy(daggerable=True)
    @no_type_check
    def prepare(qs: array[qubit, 2]) -> None:
        ry(qs[0], angle(theta))

    @guppy(unitary=True)
    @no_type_check
    def select(qs: array[qubit, 2]) -> None:
        x(qs[0])
        cx(qs[0], qs[1])
        x(qs[0])
        cz(qs[0], qs[1])

    prepare_hamiltonian = zqp.RealTermSum.from_str("(1.0, Y0)", 2)
    prepare_matrix = _hamiltonian_unitary(prepare_hamiltonian, theta)
    # |0><0| tensor X + |1><1| tensor Z.
    select_hamiltonian = zqp.RealTermSum.from_str(
        "(0.5, X1), (0.5, Z0 X1), (0.5, Z1), (-0.5, Z0 Z1)", 2
    )
    select_matrix = select_hamiltonian.to_sparse_matrix(big_endian=True).toarray()
    return ConjugationCase(
        g=prepare,
        u=select,
        width=2,
        matrix=prepare_matrix.conj().T @ select_matrix @ prepare_matrix,
    )


@pytest.mark.parametrize("theta", [0.37, -0.71])
@pytest.mark.parametrize(
    "make_case",
    [
        pytest.param(_basis_rotation_case, id="basis_rotation"),
        pytest.param(_parallel_rotations_case, id="parallel_rotations"),
        pytest.param(_pauli_gadget_case, id="pauli_gadget_zzz"),
        pytest.param(_zx_pauli_gadget_case, id="pauli_gadget_zx"),
        pytest.param(_xyz_pauli_gadget_case, id="pauli_gadget_xyz"),
        pytest.param(_two_term_lcu_case, id="two_term_lcu"),
    ],
)
def test_nested_compose_modifiers(
    make_case: Callable[[float], ConjugationCase], theta: float
) -> None:
    """Check all four modifiers for each conjugation box and input angle."""
    case = make_case(theta)
    g, u, width = case.g, case.u, case.width

    @guppy(unitary=True)
    @no_type_check
    def circuit(qs: array[qubit, width]) -> None:
        box = ConjugationBox(g, u)
        box.compose(qs)

    assert_unitary_modifiers(circuit, width, expected_unitary=case.matrix)
