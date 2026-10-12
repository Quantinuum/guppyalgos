"""Tests for the controlled qubitization walk operator."""

from typing import no_type_check

import pytest
import zixy.qubit.pauli as zqp
from guppylang import guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.builtins import array, nat
from guppylang.std.quantum import h, qubit
from pytest_lazy_fixtures import lf as lazy_fixture

from guppyalgos.primitives.gate_decompositions.cnx.cnx import cnx
from guppyalgos.algorithms.block_encoding.lcu import (
    LCUCntrl,
    LCUData,
    build_cntrl_single_cntrl_select,
    build_cntrl_unary_iteration_select,
)
from guppyalgos.algorithms.block_encoding.qubitization import QubitizationCntrl
from guppyalgos.primitives.subroutines.reflection import ReflectionCntrl
from guppyalgos.algorithms.state_preparation import multiplexor_prep
from guppyalgos.testing import (
    Endianness,
    assert_cntrl_unitary,
    chebyshev_power_matrix,
)

dagger = object()


def _build_cntrl_qubitization[n_prep_q: nat, n_state_q: nat](
    data: LCUData,
    prepare: GuppyFunctionDefinition[[array[qubit, n_prep_q]], None],
    cntrl_select: GuppyFunctionDefinition[
        [qubit, array[qubit, n_prep_q], array[qubit, n_state_q]], None
    ],
    power: int,
) -> GuppyFunctionDefinition[
    [qubit, array[qubit, n_prep_q], array[qubit, n_state_q]], None
]:
    """Build a controlled qubitization walk raised to the given power."""
    n_prep_qubits = data.n_prep_qubits
    n_state_qubits = data.n_state_qubits

    @guppy
    @no_type_check
    def unprepare(prep: array[qubit, n_prep_qubits]) -> None:
        with dagger:
            prepare(prep)

    @guppy
    @no_type_check
    def main(
        control: qubit,
        prep: array[qubit, n_prep_qubits],
        state: array[qubit, n_state_qubits],
    ) -> None:
        cntrl_lcu = LCUCntrl(prepare, cntrl_select, unprepare)
        reflection = ReflectionCntrl[n_prep_qubits](cnx)
        QubitizationCntrl(cntrl_lcu, reflection).power(control, prep, state, power)

    return main


def _assert_cntrl_qubitization_encodes_chebyshev[n_prep_q: nat, n_state_q: nat](
    ham_op: zqp.RealTermSum,
    data: LCUData,
    prepare: GuppyFunctionDefinition[[array[qubit, n_prep_q]], None],
    cntrl_select: GuppyFunctionDefinition[
        [qubit, array[qubit, n_prep_q], array[qubit, n_state_q]], None
    ],
    power: int,
    *,
    n_extra_qubits: int = 0,
) -> None:
    r"""Check the coherently interfered blocks of controlled :math:`W^k`.

    Construct a controlled LCU from ``prepare``, ``cntrl_select``, and the
    inverse of ``prepare``, then combine it with a controlled all-zero
    reflection. The resulting walk applies the identity when the external
    control is zero and :math:`W^k` when it is one.

    Hadamards before and after the controlled walk test coherence between these
    two branches.
    After projecting the PREPARE register onto its all-zero state, external
    control outcomes of zero and one must encode

    $$
    \frac{I + (-1)^k T_k(H / \lambda)}{2}
    \quad\text{and}\quad
    \frac{I - (-1)^k T_k(H / \lambda)}{2},
    $$

    respectively. The factor :math:`(-1)^k` follows from the reflection
    convention :math:`R = I - 2|0\rangle\!\langle 0|`. Both blocks are stacked
    before phase alignment so that only one global phase is ignored; their
    relative phase remains part of the assertion.

    Args:
        ham_op: Hermitian Hamiltonian encoded by the LCU.
        data: Classical term, coefficient, and register-size data for
            ``ham_op``.
        prepare: PREPARE oracle for the LCU coefficient amplitudes.
        cntrl_select: SELECT oracle enabled by the external control.
        power: Positive walk-operator power :math:`k`.
        n_extra_qubits: Simulator capacity for work qubits allocated inside
            SELECT.

    """
    n_prep_qubits = data.n_prep_qubits
    n_state_qubits = data.n_state_qubits
    ctrl_qubitization = _build_cntrl_qubitization(data, prepare, cntrl_select, power)

    @guppy
    @no_type_check
    def cntrl_qubitization(
        control: array[qubit, 1],
        prep: array[qubit, n_prep_qubits],
        state: array[qubit, n_state_qubits],
    ) -> None:
        h(control[0])
        ctrl_qubitization(control[0], prep, state)
        h(control[0])

    normalized_hamiltonian = ham_op.to_sparse_matrix(False).toarray() / data.l1_norm
    chebyshev = (-1) ** power * chebyshev_power_matrix(normalized_hamiltonian, power)
    assert_cntrl_unitary(
        cntrl_qubitization,
        chebyshev,
        n_state_qubits,
        {"prep": [False] * n_prep_qubits},
        n_extra_qubits=n_extra_qubits,
        endianness=Endianness.LITTLE,
    )


@pytest.mark.parametrize(
    "ham_op",
    [
        lazy_fixture("ham_1q_posreal_0"),
        lazy_fixture("ham_1q_posreal_4"),
        lazy_fixture("ham_1q_negreal_0"),
        lazy_fixture("ham_1q_mixedreal_0"),
    ],
)
@pytest.mark.parametrize("power", [1, 2, 3])
def test_cntrl_qubitization_single_cntrl_select(
    ham_op: zqp.RealTermSum,
    power: int,
) -> None:
    """Test controlled qubitization using single-control SELECT."""
    data = LCUData.from_hamiltonian(ham_op)
    prepare = multiplexor_prep(data.amplitudes)
    cntrl_select = build_cntrl_single_cntrl_select(data)

    assert data.n_prep_qubits == 1
    _assert_cntrl_qubitization_encodes_chebyshev(
        ham_op,
        data,
        prepare,
        cntrl_select,
        power,
    )


@pytest.mark.parametrize(
    "ham_op",
    [
        lazy_fixture("ham_2q_mixedreal_0"),
        lazy_fixture("ham_3q_posreal_0"),
        lazy_fixture("ham_3q_posreal_1"),
        lazy_fixture("ham_3q_negreal_0"),
    ],
)
@pytest.mark.parametrize("power", [1, 2, 3])
def test_cntrl_qubitization_unary_iteration_select(
    ham_op: zqp.RealTermSum,
    power: int,
) -> None:
    """Test controlled qubitization using unary-iteration SELECT."""
    data = LCUData.from_hamiltonian(ham_op)
    prepare = multiplexor_prep(data.amplitudes)
    cntrl_select = build_cntrl_unary_iteration_select(data)

    assert data.n_prep_qubits == 2
    _assert_cntrl_qubitization_encodes_chebyshev(
        ham_op,
        data,
        prepare,
        cntrl_select,
        power,
        n_extra_qubits=data.n_prep_qubits,
    )
