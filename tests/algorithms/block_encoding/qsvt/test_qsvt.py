"""Tests for the QSVT implementation."""

from typing import no_type_check

import pytest
from dataclasses import dataclass
from numpy.polynomial import Polynomial
import numpy as np
from scipy.linalg import svd
from sympy.core.expr import Expr
from numpy.typing import NDArray
from collections.abc import Sequence
import zixy.qubit.pauli as zqp

from pytest_lazy_fixtures import lf as lazy_fixture

from guppylang import guppy
from guppylang.std.builtins import array, comptime, dagger, nat
from guppylang.std.quantum import qubit
from guppylang.defs import GuppyFunctionDefinition

from guppyalgos.algorithms.block_encoding.qsvt import QSVT
from guppyalgos.testing import (
    Endianness,
    assert_allclose_ignorephase,
    get_unitary_projected,
)
from guppyalgos.algorithms.block_encoding.lcu import (
    LCU,
    LCUData,
    build_double_cntrl_select,
    build_single_cntrl_select,
    build_unary_iteration_select,
)
from guppyalgos.algorithms.state_preparation import multiplexor_prep
from guppyalgos.utils import qsp_phase_reflection


def scipy_qsvt(
    operator: NDArray[np.complex128], polynomial: Polynomial
) -> NDArray[np.complex128]:
    """Scipy implementation of QSVT.

    Polynomial transform of the singular values, where the SVD
    is performed using scipy.linalg.svd. For even polynomial,
    the right singular vectors are used, and for odd polynomial,
    the left singular vectors are used.

    Args:
    ----
        operator (npt.ndarray): matrix operator
        polynomial (Polynomial): polynomial to be applied to
        the singular values of the matrix

    """
    U, s, Vh = svd(operator, full_matrices=True)

    if (len(polynomial) - 1) % 2 == 0:
        # even polynomial
        # ∑_{k} Poly(s_k)|vk> <vk|
        # ONLY USES RIGHT SINGULAR VECS |vk>!
        qsvt = Vh.conj().T @ np.diag(polynomial(s)) @ Vh

    else:
        # odd polynomial
        # ∑_{k} Poly(s_k)|uk> <vk|
        qsvt = U @ np.diag(polynomial(s)) @ Vh

    return qsvt


@dataclass
class QSVTTestData:
    """Dataclass for QSVT tests."""

    phases: Sequence[float | Expr]
    np_poly: Polynomial
    n_it: int


@pytest.fixture
def cos_19() -> QSVTTestData:
    """Cosine polynomial transform data."""
    cos_19_odd_phases = [
        0.17475809317830504,
        -0.6144644794997792,
        0.1644479247750522,
        0.3485904836773238,
        0.16207538080468725,
        -0.8765149972894701,
        -1.0829475221847624,
        1.1535622790787932,
        1.2271523102762942,
        -0.31795256397605454,
        -1.9144430182150582,
        1.153566426080649,
        -1.0829445330474403,
        2.2650756313757827,
        -2.9795182321728144,
        0.34859056526776966,
        0.1644481777008766,
        2.5271282224435976,
        1.7455542836637912,
    ]
    cos_19_odd_phases = qsp_phase_reflection(cos_19_odd_phases)
    cos_19_odd_phases_np_poly = [
        0.9999755072730688,
        0.0,
        -49.99503640436336,
        0.0,
        416.5001907077353,
        0.0,
        -1386.7141055619081,
        0.0,
        2465.6447583388326,
        0.0,
        -2699.594131284049,
        0.0,
        1953.5740160332634,
        0.0,
        -943.8002844190673,
        0.0,
        282.50740673213994,
        0.0,
        -39.9618828784171,
    ]
    return QSVTTestData(
        cos_19_odd_phases, Polynomial(cos_19_odd_phases_np_poly), len(cos_19_odd_phases)
    )


@pytest.fixture
def sin_20() -> QSVTTestData:
    """Sine polynomial transform data."""
    qsp_phases_sin = [
        -0.6057414088269119,
        -1.173201920107398,
        1.975995776451867,
        -0.3504255865671567,
        -1.8601327504796605,
        1.3325826442509068,
        -1.1398374966618863,
        2.0017541595489194,
        -1.8090107846178523,
        1.2814605311172615,
        -0.35042514176055545,
        -1.1655968688052725,
        -1.1732019524997273,
        0.9650548924507019,
    ]
    qsp_phases_sin = qsp_phase_reflection(qsp_phases_sin)
    poly_np_sin = [
        0.0,
        2.4999926047130057,
        0.0,
        -10.416388389196523,
        0.0,
        13.017796089131378,
        0.0,
        -7.735817537599846,
        0.0,
        2.654458350152755,
        0.0,
        -0.5617930428848656,
        0.0,
        0.06229025671515937,
    ]
    return QSVTTestData(qsp_phases_sin, Polynomial(poly_np_sin), len(qsp_phases_sin))


def _get_conj_ham(
    ham_op: zqp.RealTermSum | zqp.ComplexTermSum,
) -> zqp.ComplexTermSum:
    """Return the adjoint Hamiltonian by conjugating its coefficients."""
    terms = (
        zqp.ComplexTerm.from_cmpnt_coeff(
            term.string,  # ty: ignore[unresolved-attribute]
            complex(np.conj(term.coeff)),
        )
        for term in ham_op.to_terms()
    )
    return zqp.ComplexTermSum.from_iterable(terms, ham_op.qubits)


def _test_qsvt_given_lcu[n_prep_q: nat, n_state_q: nat](
    data: LCUData,
    ham_op: zqp.RealTermSum | zqp.ComplexTermSum,
    matrix_function_data: QSVTTestData,
    prepare: GuppyFunctionDefinition[[array[qubit, n_prep_q]], None],
    select: GuppyFunctionDefinition[
        [array[qubit, n_prep_q], array[qubit, n_state_q]], None
    ],
    select_dagger: GuppyFunctionDefinition[
        [array[qubit, n_prep_q], array[qubit, n_state_q]], None
    ],
    n_extra_qubits: int = 0,
) -> None:
    """Test QSVT implementation given LCU implementation."""
    n_prep_qubits = data.n_prep_qubits
    n_state_qubits = data.n_state_qubits

    phases = matrix_function_data.phases

    @guppy
    @no_type_check
    def unprepare(prep: array[qubit, n_prep_qubits]) -> None:
        with dagger:
            prepare(prep)

    @guppy
    def qsvt_circ(
        prep_qreg: array[qubit, n_prep_qubits],
        signal_qreg: array[qubit, 1],
        select_qreg: array[qubit, n_state_qubits],
    ) -> None:
        QSVT(
            LCU(prepare, select, unprepare),
            LCU(prepare, select_dagger, unprepare),
            comptime(phases),
        ).compose(signal_qreg[0], prep_qreg, select_qreg)

    post_selection = {
        "prep_qreg": [False for _ in range(n_prep_qubits)],
        "signal": [False],
    }

    guppy_h = get_unitary_projected(
        qsvt_circ,
        n_state_qubits,
        post_selection,
        endianness=Endianness.LITTLE,
        n_extra_qubits=n_extra_qubits,
    )

    actual_hamiltonian = ham_op.to_sparse_matrix().toarray()
    actual_hamiltonian_normalized = actual_hamiltonian / data.l1_norm
    scipy_qsvt_matrix = scipy_qsvt(
        actual_hamiltonian_normalized, matrix_function_data.np_poly
    )

    assert_allclose_ignorephase(guppy_h, scipy_qsvt_matrix, threshold=1e-3)


@pytest.mark.parametrize(
    "ham_op",
    [
        lazy_fixture("op_fixture"),
    ],
)
@pytest.mark.parametrize(
    "matrix_function_data", [lazy_fixture("sin_20"), lazy_fixture("cos_19")]
)
def test_qsvt(
    ham_op: zqp.RealTermSum | zqp.ComplexTermSum,
    matrix_function_data: QSVTTestData,
) -> None:
    """Test QSVT guppy implementation against scipy QSVT implementation."""
    data = LCUData.from_hamiltonian(ham_op)
    ham_op_dagger = _get_conj_ham(ham_op)
    data_dagger = LCUData.from_hamiltonian(ham_op_dagger)
    prepare = multiplexor_prep(data.amplitudes)

    assert data.n_prep_qubits in [1, 2], "Only 1 or 2 preparation qubits supported"

    if data.n_prep_qubits == 1:
        select = build_single_cntrl_select(data)
        select_dagger = build_single_cntrl_select(data_dagger)
    elif data.n_prep_qubits == 2:
        select = build_double_cntrl_select(data)
        select_dagger = build_double_cntrl_select(data_dagger)

    _test_qsvt_given_lcu(
        data, ham_op, matrix_function_data, prepare, select, select_dagger
    )


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
@pytest.mark.parametrize(
    "matrix_function_data", [lazy_fixture("sin_20"), lazy_fixture("cos_19")]
)
def test_qsvt_unary_select(
    ham_op: zqp.RealTermSum | zqp.ComplexTermSum,
    matrix_function_data: QSVTTestData,
) -> None:
    """Test QSVT guppy against scipy QSVT implementation using unary select."""
    data = LCUData.from_hamiltonian(ham_op)
    ham_op_dagger = _get_conj_ham(ham_op)
    data_dagger = LCUData.from_hamiltonian(ham_op_dagger)
    prepare = multiplexor_prep(data.amplitudes)
    select = build_unary_iteration_select(data)
    select_dagger = build_unary_iteration_select(data_dagger)

    n_extra_qubits = data.n_prep_qubits - 1

    _test_qsvt_given_lcu(
        data,
        ham_op,
        matrix_function_data,
        prepare,
        select,
        select_dagger,
        n_extra_qubits=n_extra_qubits,
    )
