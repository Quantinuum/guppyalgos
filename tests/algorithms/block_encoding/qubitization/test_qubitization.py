"""Tests for the qubitization (walk operator) block encoding."""

from typing import no_type_check

import pytest
import zixy.qubit.pauli as zqp
from guppylang import guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.builtins import array, nat
from guppylang.std.quantum import qubit
from pytest_lazy_fixtures import lf as lazy_fixture

from guppyalgos.primitives.gate_decompositions.cnx.cnx import cnx
from guppyalgos.algorithms.block_encoding.lcu import (
    LCU,
    LCUData,
    build_double_cntrl_select,
    build_single_cntrl_select,
    build_unary_iteration_select,
)
from guppyalgos.algorithms.block_encoding.qubitization.qubitization import Qubitization
from guppyalgos.primitives.subroutines.reflection import Reflection
from guppyalgos.algorithms.state_preparation import multiplexor_prep
from guppyalgos.testing import (
    Endianness,
    assert_allclose_ignorephase,
    chebyshev_power_matrix,
    get_unitary_projected,
)

dagger = object()


def _build_qubitization[n_prep_q: nat, n_state_q: nat, n_control_q: nat](
    data: LCUData,
    prepare: GuppyFunctionDefinition[[array[qubit, n_prep_q]], None],
    select: GuppyFunctionDefinition[
        [array[qubit, n_prep_q], array[qubit, n_state_q]], None
    ],
    power: int,
    n_control_qubits: int,
) -> GuppyFunctionDefinition[[array[qubit, n_prep_q], array[qubit, n_state_q]], None]:
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
        prep_qreg: array[qubit, n_prep_qubits],
        select_qreg: array[qubit, n_state_qubits],
    ) -> None:
        lcu_struct = LCU(prepare, select, unprepare)
        reflection_struct = Reflection[n_prep_qubits, n_control_qubits](cnx)
        qubitize = Qubitization(lcu_struct, reflection_struct)
        qubitize.power(prep_qreg, select_qreg, power)

    return main


def _assert_qubitization_encodes_chebyshev(
    ham_op: zqp.RealTermSum,
    data: LCUData,
    prepare: GuppyFunctionDefinition,
    select: GuppyFunctionDefinition,
    power: int,
    *,
    n_control_qubits: int,
) -> None:
    qubitization = _build_qubitization(data, prepare, select, power, n_control_qubits)
    guppy_hamiltonian = get_unitary_projected(
        qubitization,
        data.n_state_qubits,
        {"prep_qreg": [False] * data.n_prep_qubits},
        n_extra_qubits=n_control_qubits,
        endianness=Endianness.LITTLE,
    )

    actual_hamiltonian = ham_op.to_sparse_matrix(False).toarray()
    actual_hamiltonian_normalized = actual_hamiltonian / data.l1_norm
    actual_chebyshev = chebyshev_power_matrix(actual_hamiltonian_normalized, power)
    assert_allclose_ignorephase(guppy_hamiltonian, actual_chebyshev)


@pytest.mark.parametrize(
    "ham_op",
    [
        lazy_fixture("ham_1q_posreal_0"),
        lazy_fixture("ham_1q_posreal_1"),
        lazy_fixture("ham_1q_posreal_2"),
        lazy_fixture("ham_1q_posreal_3"),
        lazy_fixture("ham_1q_posreal_4"),
        lazy_fixture("ham_1q_posreal_5"),
        lazy_fixture("ham_1q_posreal_6"),
        lazy_fixture("ham_1q_negreal_0"),
        lazy_fixture("ham_1q_mixedreal_0"),
        lazy_fixture("ham_2q_posreal_0"),
        lazy_fixture("ham_2q_posreal_1"),
        lazy_fixture("ham_2q_posreal_2"),
        lazy_fixture("ham_2q_posreal_3"),
        lazy_fixture("ham_2q_posreal_4"),
        lazy_fixture("ham_2q_negreal_0"),
        lazy_fixture("ham_2q_negreal_1"),
    ],
)
@pytest.mark.parametrize("power", [1, 2, 3])
def test_qubitization_single_cntrl_select(
    ham_op: zqp.RealTermSum,
    power: int,
) -> None:
    """Test qubitization using single-control SELECT."""
    data = LCUData.from_hamiltonian(ham_op)
    prepare = multiplexor_prep(data.amplitudes)
    select = build_single_cntrl_select(data)

    assert data.n_prep_qubits == 1
    _assert_qubitization_encodes_chebyshev(
        ham_op,
        data,
        prepare,
        select,
        power,
        n_control_qubits=0,
    )


@pytest.mark.parametrize(
    "ham_op",
    [
        lazy_fixture("ham_2q_mixedreal_0"),
        lazy_fixture("ham_3q_posreal_0"),
        lazy_fixture("ham_3q_posreal_1"),
        lazy_fixture("ham_3q_negreal_0"),
        lazy_fixture("ham_3q_negreal_1"),
    ],
)
@pytest.mark.parametrize("power", [1, 2, 3])
def test_qubitization_double_cntrl_select(
    ham_op: zqp.RealTermSum,
    power: int,
) -> None:
    """Test qubitization using double-control SELECT."""
    data = LCUData.from_hamiltonian(ham_op)
    prepare = multiplexor_prep(data.amplitudes)
    select = build_double_cntrl_select(data)

    assert data.n_prep_qubits == 2
    _assert_qubitization_encodes_chebyshev(
        ham_op,
        data,
        prepare,
        select,
        power,
        n_control_qubits=1,
    )


@pytest.mark.parametrize(
    "ham_op",
    [
        lazy_fixture("ham_2q_mixedreal_0"),
        lazy_fixture("ham_3q_posreal_0"),
        lazy_fixture("ham_3q_posreal_1"),
        lazy_fixture("ham_3q_negreal_0"),
        lazy_fixture("ham_3q_negreal_1"),
    ],
)
@pytest.mark.parametrize("power", [1, 2, 3])
def test_qubitization_unary_iteration_select(
    ham_op: zqp.RealTermSum,
    power: int,
) -> None:
    """Test qubitization using unary-iteration SELECT."""
    data = LCUData.from_hamiltonian(ham_op)
    prepare = multiplexor_prep(data.amplitudes)
    select = build_unary_iteration_select(data)

    assert data.n_prep_qubits == 2
    _assert_qubitization_encodes_chebyshev(
        ham_op,
        data,
        prepare,
        select,
        power,
        n_control_qubits=data.n_prep_qubits - 1,
    )
