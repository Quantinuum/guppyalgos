"""Tests for the externally controlled LCU block encoding."""

from typing import no_type_check

import pytest
import zixy.qubit.pauli as zqp
from guppylang import guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.builtins import array, nat
from guppylang.std.quantum import h, qubit
from pytest_lazy_fixtures import lf as lazy_fixture

from guppyalgos.primitives.gate_decompositions.and_op import (
    temp_and_compute,
    temp_and_uncompute,
)
from guppyalgos.algorithms.block_encoding.lcu import (
    LCUCntrl,
    LCUData,
    build_cntrl_single_cntrl_select,
    build_cntrl_unary_iteration_select,
)
from guppyalgos.algorithms.state_preparation import multiplexor_prep
from guppyalgos.testing import Endianness, assert_cntrl_unitary

dagger = object()


def _assert_cntrl_lcu_encodes_hamiltonian[n_prep_q: nat, n_state_q: nat](
    ham_op: zqp.RealTermSum | zqp.ComplexTermSum,
    data: LCUData,
    prepare: GuppyFunctionDefinition[[array[qubit, n_prep_q]], None],
    cntrl_select: GuppyFunctionDefinition[
        [qubit, array[qubit, n_prep_q], array[qubit, n_state_q]], None
    ],
    *,
    n_extra_qubits: int = 0,
) -> None:
    """Check both coherently interfered control blocks of a controlled LCU."""
    n_prep_qubits = data.n_prep_qubits
    n_state_qubits = data.n_state_qubits

    @guppy
    @no_type_check
    def unprepare(prep: array[qubit, n_prep_qubits]) -> None:
        with dagger:
            prepare(prep)

    @guppy
    @no_type_check
    def cntrl_lcu(
        control: array[qubit, 1],
        prep: array[qubit, n_prep_qubits],
        state: array[qubit, n_state_qubits],
    ) -> None:
        h(control[0])
        LCUCntrl(prepare, cntrl_select, unprepare).compose(control[0], prep, state)
        h(control[0])

    normalized_hamiltonian = ham_op.to_sparse_matrix().toarray() / data.l1_norm
    assert_cntrl_unitary(
        cntrl_lcu,
        normalized_hamiltonian,
        n_state_qubits,
        {"prep": [False] * n_prep_qubits},
        endianness=Endianness.LITTLE,
        n_extra_qubits=n_extra_qubits,
    )


@pytest.mark.parametrize(
    "ham_op",
    [
        lazy_fixture("ham_1q_mixedreal_0"),
        lazy_fixture("ham_1q_posimaginary_0"),
        lazy_fixture("ham_2q_mixed_0"),
    ],
)
def test_cntrl_lcu_single_cntrl_select(
    ham_op: zqp.RealTermSum | zqp.ComplexTermSum,
) -> None:
    r"""Test both control branches and their relative phase for two-term LCU.

    Hadamards before and after the external control coherently interfere the
    inactive identity branch with the active LCU branch. After projecting the
    PREPARE register onto zero, control outcomes zero and one must respectively
    encode :math:`(I + H/\lambda)/2` and :math:`(I - H/\lambda)/2`.

    """
    data = LCUData.from_hamiltonian(ham_op)
    prepare = multiplexor_prep(data.amplitudes)
    cntrl_select = build_cntrl_single_cntrl_select(data)
    assert data.n_prep_qubits == 1
    _assert_cntrl_lcu_encodes_hamiltonian(ham_op, data, prepare, cntrl_select)


@pytest.mark.parametrize(
    "ham_op",
    [
        lazy_fixture("ham_2q_mixedreal_0"),
        lazy_fixture("ham_3q_mixed_0"),
    ],
)
def test_cntrl_lcu_unary_iteration_select(
    ham_op: zqp.RealTermSum | zqp.ComplexTermSum,
) -> None:
    """Test controlled unary-iteration LCU for three and four terms."""
    data = LCUData.from_hamiltonian(ham_op)
    prepare = multiplexor_prep(data.amplitudes)
    cntrl_select = build_cntrl_unary_iteration_select(data)

    assert data.n_prep_qubits == 2
    _assert_cntrl_lcu_encodes_hamiltonian(
        ham_op,
        data,
        prepare,
        cntrl_select,
        n_extra_qubits=data.n_prep_qubits,
    )


def test_cntrl_lcu_unary_iteration_select_custom_and_ops(
    ham_2q_mixedreal_0: zqp.RealTermSum,
) -> None:
    """Test controlled unary-iteration SELECT with temporary AND operations."""
    data = LCUData.from_hamiltonian(ham_2q_mixedreal_0)
    prepare = multiplexor_prep(data.amplitudes)
    cntrl_select = build_cntrl_unary_iteration_select(
        data,
        comp_and_op=temp_and_compute,
        uncomp_and_op=temp_and_uncompute,
    )

    _assert_cntrl_lcu_encodes_hamiltonian(
        ham_2q_mixedreal_0,
        data,
        prepare,
        cntrl_select,
        n_extra_qubits=data.n_prep_qubits,
    )
