"""QFT ladder tests."""

import pytest
import numpy as np
from typing import no_type_check
from guppylang.decorator import guppy
from guppylang.std.builtins import array
from guppylang.std.quantum import qubit
from selene_sim import Quest


from guppylang.std.debug import state_output

from numpy.typing import NDArray

from guppyalgos.primitives.subroutines.qft import qft, iqft
from guppylang.std.quantum import x
from guppylang.std.quantum import (
    h,
    discard_array,
)
from guppyalgos.primitives.state_preparation.uniform import uniform_state
from guppyalgos.testing import (
    assert_allclose_ignorephase,
    get_statevector,
    Endianness,
    get_unitary_assumed_phase,
    switch_endianness,
)
from guppyalgos.utils import qarray


def qft_unitary_algebra(n_qubits: int) -> NDArray[np.complex128]:
    """Return the unitary matrix for the n qubit Quantum Fourier transform."""
    N = 2**n_qubits
    omega = np.exp(2j * np.pi / N)
    QFT = np.zeros((N, N), dtype=complex)

    for j in range(N):
        for k in range(N):
            QFT[j, k] = omega ** (j * k)

    return QFT / np.sqrt(N)


@pytest.mark.parametrize("n_qubits", [1, 3, 4])
@no_type_check
def test_qft_unitary(n_qubits: int) -> None:
    """Test the QFT unitary matrix generation."""
    qft_arr = qft_unitary_algebra(n_qubits)

    # Verify unitarity: U * U† = I
    identity = np.eye(2**n_qubits)
    product = np.dot(qft_arr, qft_arr.conj().T)

    assert_allclose_ignorephase(product, identity)

    if n_qubits == 1:

        @guppy
        def main(state_qreg: array[qubit, n_qubits]) -> None:
            h(state_qreg[0])

        guppy_u = get_unitary_assumed_phase(
            main, n_qubits, endianness=Endianness.LITTLE
        )
        assert_allclose_ignorephase(guppy_u, qft_arr.conj())

    @guppy
    def main2(state_qreg: array[qubit, n_qubits]) -> None:
        iqft(state_qreg)

    guppy_iQFT = get_unitary_assumed_phase(
        main2, n_qubits, endianness=Endianness.LITTLE
    )
    assert_allclose_ignorephase(guppy_iQFT, qft_arr.conj())

    @guppy
    def main3(state_qreg: array[qubit, n_qubits]) -> None:
        qft(state_qreg)

    guppy_QFT = get_unitary_assumed_phase(main3, n_qubits, endianness=Endianness.LITTLE)
    assert_allclose_ignorephase(guppy_QFT, qft_arr)


@pytest.mark.parametrize("n_qubits", [1, 3, 4])
@no_type_check
def test_iqft_sv(n_qubits: int) -> None:
    """Test the IQFT U by transforming a uniform superposition to |0>."""
    uniform_state_prep = uniform_state(2**n_qubits)

    @guppy
    def main() -> None:
        qs = qarray(n_qubits)
        uniform_state_prep(qs)
        iqft(qs)
        state_output("result_state", qs)
        discard_array(qs)

    simulated_state = get_statevector(main, n_qubits)
    expected_state = np.zeros(simulated_state.shape, dtype=complex)
    expected_state[0] = 1.0
    assert_allclose_ignorephase(simulated_state, expected_state)


@pytest.mark.parametrize("n_qubits", [1, 3, 4])
@no_type_check
def test_qft_sv(n_qubits: int) -> None:
    """Test the QFT U by transforming |0> to a uniform superposition."""

    @guppy
    def main2() -> None:
        qs = qarray(n_qubits)
        qft(qs)
        state_output("result_state", qs)
        discard_array(qs)

    expected_state = np.array([1 / (2 ** (n_qubits / 2))] * (2**n_qubits))
    simulated_state = get_statevector(main2, n_qubits)
    assert_allclose_ignorephase(simulated_state, expected_state)


@pytest.mark.parametrize("n_qubits", [1, 3, 4])
@no_type_check
def test_qft_roundtrip_unitary(n_qubits: int) -> None:
    """Test QFT iQFT produces identity."""

    @guppy
    def main(state_qreg: array[qubit, n_qubits]) -> None:
        qft(state_qreg)
        iqft(state_qreg)

    U = get_unitary_assumed_phase(main, n_qubits, endianness=Endianness.LITTLE)
    identity = np.eye(2**n_qubits)
    assert_allclose_ignorephase(U, identity)


@pytest.mark.parametrize("n_qubits", [3, 4])
@pytest.mark.parametrize("k", [1, 3, 5])
@no_type_check
def test_qft_basis_state(n_qubits: int, k: int) -> None:
    """Test that QFT correctly transforms a computational basis state |k⟩."""
    N = 2**n_qubits
    omega = np.exp(2j * np.pi / N)

    @guppy
    def main() -> None:
        qs = qarray(n_qubits)
        for i in range(n_qubits):
            # little-endian bit check
            if (k >> i) & 1:
                x(qs[i])

        qft(qs)
        state_output("result_state", qs)
        discard_array(qs)

    simulated = get_statevector(main, n_qubits)

    expected = np.array(
        [omega ** (j * k) for j in range(N)],
        dtype=complex,
    ) / np.sqrt(N)
    expected = switch_endianness(expected)

    assert_allclose_ignorephase(simulated, expected)


@pytest.mark.parametrize("n_qubits", [3, 4])
@no_type_check
def test_random_state_roundtrip(n_qubits: int) -> None:
    """Check QFT|iQFT recovers Psi."""
    from pytket import Circuit
    from pytket.circuit import StatePreparationBox
    from pytket.passes import DecomposeBoxes

    N = 2**n_qubits

    # Prepare a random normalized complex vector
    rng = np.random.default_rng(123)
    psi = rng.normal(size=N) + 1j * rng.normal(size=N)
    psi = psi / np.linalg.norm(psi)

    # ---- Build circuit for the random state ----
    circ = Circuit(n_qubits)
    prep_box = StatePreparationBox(psi)
    circ.add_state_preparation_box(prep_box, list(range(n_qubits)))
    DecomposeBoxes().apply(circ)
    sp_func = guppy.load_pytket("rand_state_prep", circ)
    assert np.allclose(psi, circ.get_statevector())

    @guppy
    def main() -> None:
        qs = qarray(n_qubits)
        sp_func(qs)
        qft(qs)
        iqft(qs)
        state_output("result_state", qs)
        discard_array(qs)

    res = main.emulator(n_qubits).run()
    states = Quest.extract_states_dict(res.results[0].entries)
    simulated = states["result_state"].get_single_state()
    # TODO: get_statevector fails here due to selene qubit label issues
    # simulated = get_statevector(main, n_qubits)
    assert_allclose_ignorephase(simulated, psi)
    # TODO: it would be nice if this also preserved global phase
    # assert(np.allclose(simulated, psi))
