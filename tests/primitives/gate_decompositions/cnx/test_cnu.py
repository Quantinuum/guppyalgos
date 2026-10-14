"""Unitary tests for multi-controlled functions, excluding clean work qubits."""

from typing import no_type_check

import numpy as np
import pytest
from guppylang import guppy
from guppylang.std.builtins import Function, array, nat
from guppylang.std.qsystem.random import RNG
from guppylang.std.quantum import angle, cx, qubit, ry

from guppyalgos.primitives.gate_decompositions.cnx.cnu import MultiCntrlFunction
from guppyalgos.primitives.gate_decompositions.cnx.cnx import cnx
from guppyalgos.primitives.gate_decompositions.cnx.cnx_approx import cnx_approx
from guppyalgos.primitives.gate_decompositions.cnx.cnx_cca import cnx_cca_logdepth_dirty
from selene_sim import QuantumReplay, Quest

from guppyalgos.testing import (
    assert_allclose_ignorephase,
    get_unitary,
)
from tests.primitives.gate_decompositions.cnx.test_cnx import (
    CnxMethod,
    cnx_cca_logdepth,
    cnx_single_ancilla,
    cnx_teleportation,
    cnx_toffoli_ladder,
    _get_num_ancillas_cnx_teleportation,
)


@guppy
@no_type_check
def controlled_ry(control: qubit, target: qubit) -> None:
    """Apply CRY(pi/3) using two CX gates."""
    ry(target, angle(1.0 / 6.0))
    cx(control, target)
    ry(target, angle(-1.0 / 6.0))
    cx(control, target)


@guppy
@no_type_check
def cnx_cca_with_clean_borrowed[n_controls: nat](
    controls: array[qubit, n_controls], target: qubit
) -> None:
    """Test dirty-CNX composition with clean borrowed qubits."""
    borrowed_a = qubit()
    borrowed_b = qubit()
    cnx_cca_logdepth_dirty(controls, target, borrowed_a, borrowed_b)
    borrowed_a.discard()
    borrowed_b.discard()


def _controlled_ry_unitary(n_controls: int) -> np.ndarray:
    unitary = np.eye(2 ** (n_controls + 1), dtype=complex)
    c, s = np.cos(np.pi / 6), np.sin(np.pi / 6)
    unitary[-2:, -2:] = [[c, -s], [s, c]]
    return unitary


@guppy
@no_type_check
def _circuit[n_controls: nat, n_qubits: nat](
    qs: array[qubit, n_qubits],
    kernel: Function[[array[qubit, n_controls], qubit], None],
) -> None:
    """Compose controlled RY with the supplied CNX kernel."""
    controls = array(qs.take(i) for i in range(n_controls))
    operation = MultiCntrlFunction[n_controls, qubit](controlled_ry, kernel)
    operation.compose(controls, qs[n_controls])
    i = 0
    for control in controls:
        qs.put(control, i)
        i += 1


@pytest.mark.parametrize("n_controls", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize(
    "method",
    [
        CnxMethod(cnx, lambda n: max(0, (n - 1) // 2)),
        CnxMethod(cnx_toffoli_ladder, lambda n: max(0, n - 2)),
        CnxMethod(cnx_single_ancilla, lambda _: 1),
        CnxMethod(cnx_cca_logdepth, lambda n: 1 if n < 6 else 2),
        CnxMethod(cnx_cca_with_clean_borrowed, lambda _: 2),
    ],
    ids=["cnx", "ladder", "single_ancilla", "cca", "cca_dirty"],
)
def test_multicontrolled_function_unitary(method: CnxMethod, n_controls: int) -> None:
    """Check the full unitary with internal work qubits removed."""
    n_extra = 0 if n_controls == 1 else 1 + method.ancilla_func(n_controls)
    n_qubits = n_controls + 1
    kernel = method.cnx_func

    @guppy
    @no_type_check
    def circuit(qs: array[qubit, n_qubits]) -> None:
        _circuit[n_controls, n_qubits](qs, kernel)

    actual = get_unitary(circuit, n_qubits, n_extra_qubits=n_extra)
    assert_allclose_ignorephase(actual, _controlled_ry_unitary(n_controls))


@pytest.mark.parametrize("n_controls", [1, 2, 3, 4, 5, 6])
def test_multicontrolled_function_teleportation(n_controls: int) -> None:
    """Check composition using the original teleportation CNX and get_unitary.

    QuantumReplay postselects every teleportation measurement onto zero while
    QuEST simulates the quantum state. Replaying the same branch for every
    input keeps discarded work slots at zero and avoids different sampled
    global phases between unitary columns. This is so it is compatible with
    get_unitary(). This test only checks composition
    on the all-zero branch; teleportation is tested separately in test_cnx.
    The replay simulator is passed directly to get_unitary; CNX is unchanged.
    """
    n_qubits = n_controls + 1
    n_ancillas = _get_num_ancillas_cnx_teleportation(n_controls)
    n_extra = 0 if n_controls == 1 else 1 + n_ancillas
    # compose invokes CNX twice; each invocation measures each ancilla once.
    n_measurements = 2 * n_ancillas if n_controls > 2 else 0
    replay = QuantumReplay(simulator=Quest(), measurements=[[False] * n_measurements])

    @guppy
    @no_type_check
    def circuit(qs: array[qubit, n_qubits]) -> None:
        _circuit[n_controls, n_qubits](qs, cnx_teleportation)

    actual = get_unitary(
        circuit,
        n_qubits,
        n_extra_qubits=n_extra,
        simulator=replay,
    )
    assert_allclose_ignorephase(actual, _controlled_ry_unitary(n_controls))


@pytest.mark.parametrize("n_controls", [6])
@pytest.mark.parametrize("seed", [1, 7])
def test_multicontrolled_function_approx_unitary(n_controls: int, seed: int) -> None:
    """Check exact fallback and genuine approximation with replayed randomness."""
    epsilon = 0.25
    approximate = cnx_approx(n_controls, epsilon)

    @guppy
    @no_type_check
    def kernel(controls: array[qubit, n_controls], target: qubit) -> None:
        rng = RNG(seed)
        approximate(controls, target, rng)
        rng.discard()

    # k=4 for epsilon=1/4; the kernel uses at most one internal ancilla.
    n_qubits = n_controls + 1

    @guppy
    @no_type_check
    def circuit(qs: array[qubit, n_qubits]) -> None:
        _circuit[n_controls, n_qubits](qs, kernel)

    actual = get_unitary(circuit, n_qubits, n_extra_qubits=2)
    np.testing.assert_allclose(
        actual.conj().T @ actual, np.eye(len(actual)), atol=1e-10
    )
    expected = _controlled_ry_unitary(n_controls)

    # For a fixed seed the approximate predicate is deterministic. Every
    # control block must be either identity or RY.
    rotation = expected[-2:, -2:]
    for i in range(2**n_controls):
        # Extract the 2x2 block corresponding to the current control state.
        block = actual[2 * i : 2 * i + 2, 2 * i : 2 * i + 2]
        expected_block = np.eye(2) if abs(block[0, 1]) < 1e-10 else rotation
        assert_allclose_ignorephase(block, expected_block, threshold=1e-10)
