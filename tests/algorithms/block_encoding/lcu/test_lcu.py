"""Tests for the LCU block encoding."""

from typing import no_type_check

import pytest
import zixy.qubit.pauli as zqp
from guppylang import guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.builtins import array, nat
from guppylang.std.quantum import qubit
from pytest_lazy_fixtures import lf as lazy_fixture

from guppyalgos.primitives.gate_decompositions.and_op import (
    temp_and_compute,
    temp_and_uncompute,
)
from guppyalgos.testing import (
    get_unitary_projected,
    assert_allclose_ignorephase,
    Endianness,
)

from guppyalgos.algorithms.block_encoding.lcu import (
    LCU,
    LCUData,
    build_double_cntrl_select,
    build_single_cntrl_select,
    build_unary_iteration_select,
)
from guppyalgos.algorithms.state_preparation import multiplexor_prep

dagger = object()


def _assert_lcu_encodes_hamiltonian[n_prep_q: nat, n_state_q: nat](
    ham_op: zqp.RealTermSum | zqp.ComplexTermSum,
    data: LCUData,
    prepare: GuppyFunctionDefinition[[array[qubit, n_prep_q]], None],
    select: GuppyFunctionDefinition[
        [array[qubit, n_prep_q], array[qubit, n_state_q]], None
    ],
    *,
    n_extra_qubits: int = 0,
) -> None:
    n_prep_qubits = data.n_prep_qubits
    n_state_qubits = data.n_state_qubits

    @guppy
    @no_type_check
    def unprepare(prep: array[qubit, n_prep_qubits]) -> None:
        with dagger:
            prepare(prep)

    @guppy
    @no_type_check
    def lcu(
        prep: array[qubit, n_prep_qubits],
        state: array[qubit, n_state_qubits],
    ) -> None:
        LCU(prepare, select, unprepare).compose(prep, state)

    guppy_hamiltonian = get_unitary_projected(
        lcu,
        data.n_state_qubits,
        {"prep": [False] * data.n_prep_qubits},
        n_extra_qubits=n_extra_qubits,
        endianness=Endianness.LITTLE,
    )

    actual_hamiltonian = ham_op.to_sparse_matrix().toarray()
    actual_hamiltonian_normalized = actual_hamiltonian / data.l1_norm
    assert_allclose_ignorephase(guppy_hamiltonian, actual_hamiltonian_normalized)


@pytest.mark.parametrize(
    "ham_op",
    [
        lazy_fixture("ham_1q_posreal_0"),
        lazy_fixture("ham_1q_negreal_0"),
        lazy_fixture("ham_1q_mixedreal_0"),
        lazy_fixture("ham_1q_posimaginary_0"),
        lazy_fixture("ham_2q_posreal_0"),
        lazy_fixture("ham_2q_negreal_0"),
        lazy_fixture("ham_2q_negimaginary_0"),
        lazy_fixture("ham_2q_mixed_0"),
        lazy_fixture("ham_2q_mixed_1"),
    ],
)
def test_lcu_single_cntrl_select(
    ham_op: zqp.RealTermSum | zqp.ComplexTermSum,
) -> None:
    """Test that the single-control LCU encodes the normalized Hamiltonian."""
    data = LCUData.from_hamiltonian(ham_op)
    prepare = multiplexor_prep(data.amplitudes)
    select = build_single_cntrl_select(data)

    assert data.n_prep_qubits == 1
    _assert_lcu_encodes_hamiltonian(ham_op, data, prepare, select)


@pytest.mark.parametrize(
    "ham_op",
    [
        lazy_fixture("ham_2q_mixedreal_0"),
        lazy_fixture("ham_3q_posreal_0"),
        lazy_fixture("ham_3q_negreal_0"),
        lazy_fixture("ham_3q_posimaginary_0"),
        lazy_fixture("ham_3q_negimaginary_0"),
        lazy_fixture("ham_3q_mixed_0"),
    ],
)
def test_lcu_double_cntrl_select(
    ham_op: zqp.RealTermSum | zqp.ComplexTermSum,
) -> None:
    """Test that the double-control LCU encodes the normalized Hamiltonian."""
    data = LCUData.from_hamiltonian(ham_op)
    prepare = multiplexor_prep(data.amplitudes)
    select = build_double_cntrl_select(data)

    assert data.n_prep_qubits == 2
    _assert_lcu_encodes_hamiltonian(ham_op, data, prepare, select)


@pytest.mark.parametrize(
    "ham_op",
    [
        lazy_fixture("ham_2q_mixedreal_0"),
        lazy_fixture("ham_3q_posreal_0"),
        lazy_fixture("ham_3q_negreal_0"),
        lazy_fixture("ham_3q_posimaginary_0"),
        lazy_fixture("ham_3q_negimaginary_0"),
        lazy_fixture("ham_3q_mixed_0"),
    ],
)
def test_lcu_unary_iteration_select(
    ham_op: zqp.RealTermSum | zqp.ComplexTermSum,
) -> None:
    """Test that the unary-iteration LCU encodes the normalized Hamiltonian."""
    data = LCUData.from_hamiltonian(ham_op)
    prepare = multiplexor_prep(data.amplitudes)
    select = build_unary_iteration_select(data)

    assert data.n_prep_qubits == 2
    _assert_lcu_encodes_hamiltonian(
        ham_op,
        data,
        prepare,
        select,
        n_extra_qubits=data.n_prep_qubits - 1,
    )


def test_lcu_unary_iteration_select_custom_and_ops(
    ham_2q_mixedreal_0: zqp.RealTermSum,
) -> None:
    """Test unary-iteration SELECT with injected temporary AND operations."""
    data = LCUData.from_hamiltonian(ham_2q_mixedreal_0)
    prepare = multiplexor_prep(data.amplitudes)
    select = build_unary_iteration_select(
        data,
        comp_and_op=temp_and_compute,
        uncomp_and_op=temp_and_uncompute,
    )

    _assert_lcu_encodes_hamiltonian(
        ham_2q_mixedreal_0,
        data,
        prepare,
        select,
        n_extra_qubits=data.n_prep_qubits - 1,
    )
