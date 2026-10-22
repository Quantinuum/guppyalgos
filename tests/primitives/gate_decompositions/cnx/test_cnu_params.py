"""Unitary tests for parameterized multi-controlled functions."""

from typing import no_type_check

import numpy as np
import pytest
from guppylang import guppy
from guppylang.std.builtins import Function, array, nat
from guppylang.std.qsystem.random import RNG
from guppylang.std.quantum import angle, cx, qubit, ry

from guppyalgos.primitives.gate_decompositions.cnx.cnu import MultiCntrlFunctionParams
from guppyalgos.primitives.gate_decompositions.cnx.cnx import cnx
from guppyalgos.primitives.gate_decompositions.cnx.cnx_approx import cnx_approx
from selene_sim import QuantumReplay, Quest

from guppyalgos.utils.guppy.unsafe_borrow import (
    _unsafe_array_borrow,
    _unsafe_array_unborrow,
)
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
from tests.primitives.gate_decompositions.cnx.test_cnu import (
    cnx_cca_with_clean_borrowed,
)


@guppy
@no_type_check
def controlled_ry_params(
    control: qubit, target: qubit, params: tuple[angle, int]
) -> None:
    """Apply repeated CRY with a runtime angle and repetition count."""
    theta, count = params
    for _ in range(count):
        ry(target, theta / 2)
        cx(control, target)
        ry(target, -theta / 2)
        cx(control, target)


def _controlled_ry_unitary(n_controls: int, theta: float, count: int) -> np.ndarray:
    unitary = np.eye(2 ** (n_controls + 1), dtype=complex)
    c, s = np.cos(np.pi * theta * count / 2), np.sin(np.pi * theta * count / 2)
    unitary[-2:, -2:] = [[c, -s], [s, c]]
    return unitary


@guppy
@no_type_check
def _circuit[n_controls: nat, n_qubits: nat](
    qs: array[qubit, n_qubits],
    kernel: Function[[array[qubit, n_controls], qubit], None],
    theta: angle,
    count: int,
) -> None:
    """Compose controlled RY with the supplied CNX kernel."""
    controls = array(qs.take(i) for i in range(n_controls))
    operation = MultiCntrlFunctionParams[n_controls, qubit, tuple[angle, int]](
        controlled_ry_params, kernel
    )
    operation.compose(controls, qs[n_controls], (theta, count))
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
@pytest.mark.parametrize(("theta", "count"), [(1 / 3, 1), (-0.2, 3), (0.4, 0)])
def test_multicontrolled_function_params_unitary(
    method: CnxMethod, n_controls: int, theta: float, count: int
) -> None:
    """Check the full unitary with internal work qubits removed."""
    n_extra = 0 if n_controls == 1 else 1 + method.ancilla_func(n_controls)
    n_qubits = n_controls + 1
    kernel = method.cnx_func

    @guppy
    @no_type_check
    def circuit(qs: array[qubit, n_qubits]) -> None:
        _circuit[n_controls, n_qubits](qs, kernel, angle(theta), count)

    actual = get_unitary(circuit, n_qubits, n_extra_qubits=n_extra)
    assert_allclose_ignorephase(
        actual, _controlled_ry_unitary(n_controls, theta, count)
    )


@pytest.mark.parametrize("n_controls", [1, 2, 3, 4, 5, 6])
def test_multicontrolled_function_params_teleportation(n_controls: int) -> None:
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
        _circuit[n_controls, n_qubits](qs, cnx_teleportation, angle(1 / 3), 2)

    actual = get_unitary(
        circuit,
        n_qubits,
        n_extra_qubits=n_extra,
        simulator=replay,
    )
    assert_allclose_ignorephase(actual, _controlled_ry_unitary(n_controls, 1 / 3, 2))


@pytest.mark.parametrize("n_controls", [6])
@pytest.mark.parametrize("seed", [1, 7])
def test_multicontrolled_function_params_approx_unitary(
    n_controls: int, seed: int
) -> None:
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
        _circuit[n_controls, n_qubits](qs, kernel, angle(-0.2), 3)

    actual = get_unitary(circuit, n_qubits, n_extra_qubits=2)
    np.testing.assert_allclose(
        actual.conj().T @ actual, np.eye(len(actual)), atol=1e-10
    )
    expected = _controlled_ry_unitary(n_controls, -0.2, 3)

    # For a fixed seed the approximate predicate is deterministic. Every
    # control block must be either identity or RY.
    rotation = expected[-2:, -2:]
    for i in range(2**n_controls):
        # Extract the 2x2 block corresponding to the current control state.
        block = actual[2 * i : 2 * i + 2, 2 * i : 2 * i + 2]
        expected_block = np.eye(2) if abs(block[0, 1]) < 1e-10 else rotation
        assert_allclose_ignorephase(block, expected_block, threshold=1e-10)


@guppy
@no_type_check
def _controlled_registers(
    control: qubit,
    targets: tuple[array[qubit, 1], array[qubit, 2]],
    params: tuple[angle, int],
) -> None:
    theta, count = params
    controlled_ry_params(control, targets[0][0], (theta, count))
    controlled_ry_params(control, targets[1][0], (-theta, count))
    controlled_ry_params(control, targets[1][1], (-theta, count))


@pytest.mark.parametrize("n_controls", [1, 2, 3])
@pytest.mark.parametrize(("theta", "count"), [(0.2, 2), (-0.2, 2)])
def test_multicontrolled_function_params_registers(
    n_controls: int, theta: float, count: int
) -> None:
    """Check parameter forwarding and restoration of two borrowed registers."""
    n_qubits = n_controls + 3

    @guppy
    @no_type_check
    def circuit(qs: array[qubit, n_qubits]) -> None:
        controls = array(qs.take(i) for i in range(n_controls))
        first = array(qs.take(n_controls + i) for i in range(1))
        second = array(qs.take(n_controls + 1 + i) for i in range(2))
        targets = (_unsafe_array_borrow(first), _unsafe_array_borrow(second))
        operation = MultiCntrlFunctionParams[
            n_controls,
            tuple[array[qubit, 1], array[qubit, 2]],
            tuple[angle, int],
        ](_controlled_registers, cnx[n_controls])
        operation.compose(controls, targets, (angle(theta), count))
        first_contents, second_contents = targets
        _unsafe_array_unborrow(first, first_contents)
        _unsafe_array_unborrow(second, second_contents)
        for i in range(n_controls):
            qs.put(controls.take(i), i)
        controls.discard_all_taken()
        qs.put(first.take(0), n_controls)
        first.discard_all_taken()
        for i in range(2):
            qs.put(second.take(i), n_controls + 1 + i)
        second.discard_all_taken()

    c, s = np.cos(np.pi * theta * count / 2), np.sin(np.pi * theta * count / 2)
    rotation = np.array([[c, -s], [s, c]])
    expected = np.eye(2**n_qubits, dtype=complex)
    expected[-8:, -8:] = np.kron(rotation, np.kron(rotation.T, rotation.T))
    n_extra = 0 if n_controls == 1 else 1 + max(0, (n_controls - 1) // 2)
    actual = get_unitary(circuit, n_qubits, n_extra_qubits=n_extra)
    assert_allclose_ignorephase(actual, expected)


@guppy
@no_type_check
def _controlled_optional_x(control: qubit, target: qubit, enabled: bool) -> None:
    if enabled:
        cx(control, target)


@pytest.mark.parametrize("n_controls", [1, 3])
@pytest.mark.parametrize("enabled", [False, True])
def test_multicontrolled_function_params_scalar(n_controls: int, enabled: bool) -> None:
    """Accept a scalar parameter without requiring a tuple bundle."""
    n_qubits = n_controls + 1

    @guppy
    @no_type_check
    def circuit(qs: array[qubit, n_qubits]) -> None:
        controls = array(qs.take(i) for i in range(n_controls))
        operation = MultiCntrlFunctionParams[n_controls, qubit, bool](
            _controlled_optional_x, cnx[n_controls]
        )
        operation.compose(controls, qs[n_controls], enabled)
        for i in range(n_controls):
            qs.put(controls.take(i), i)
        controls.discard_all_taken()

    expected = np.eye(2**n_qubits, dtype=complex)
    if enabled:
        expected[-2:, -2:] = [[0, 1], [1, 0]]
    n_extra = 0 if n_controls == 1 else 1 + max(0, (n_controls - 1) // 2)
    actual = get_unitary(circuit, n_qubits, n_extra_qubits=n_extra)
    assert_allclose_ignorephase(actual, expected)
