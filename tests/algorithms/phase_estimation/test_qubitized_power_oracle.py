"""Phase-sensitive tests for powered walks and their explicit modifiers."""

from typing import no_type_check

import numpy as np
import pytest
from guppylang import guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.angles import angle
from guppylang.std.builtins import Unitary, array, comptime, control, dagger, nat
from guppylang.std.quantum import discard_array, h, qubit, ry, rz
from guppylang.std.debug import state_output

from guppyalgos.algorithms.phase_estimation import (
    QubitizationRegs,
    qubitized_power_oracle,
)
from guppyalgos.testing import assert_allclose_ignorephase, get_statevector, get_unitary
from guppyalgos.utils import qarray
from .test_qubitized_qpe import build_qubitized_power_oracle


@guppy(unitary=True)
@no_type_check
def _walk[n_prep: nat, n_target: nat](
    intrinsic: qubit, prep: array[qubit, n_prep], target: array[qubit, n_target]
) -> None:
    with control(intrinsic):
        ry(prep[0], angle(0.31))
        with control(prep[0]):
            rz(target[0], angle(0.27))
        rz(prep[0], angle(0.43))


def _walk_matrix() -> np.ndarray:
    """Build Rz(prep) controlled-Rz(target) Ry(prep) independently."""
    theta = np.pi * 0.31 / 2
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    prep_phase = np.diag(np.exp(np.array([-1j, 1j]) * np.pi * 0.43 / 2))
    select = np.diag(
        [1, 1, np.exp(-1j * np.pi * 0.27 / 2), np.exp(1j * np.pi * 0.27 / 2)]
    )
    return np.kron(prep_phase, np.eye(2)) @ select @ np.kron(rotation, np.eye(2))


@pytest.mark.parametrize("power", [-1, 0, 1, 3])
def test_power_oracle_matrix(power: int) -> None:
    """Check both intrinsic branches, relative phases, and register restoration."""

    @guppy
    @no_type_check
    def circuit(qs: array[qubit, 3]) -> None:
        regs = QubitizationRegs(array(qs.take(1)), array(qs.take(2)))
        qubitized_power_oracle(qs[0], regs.prep, regs.target, power, _walk)
        qs.put(regs.prep.take(0), 1)
        qs.put(regs.target.take(0), 2)
        regs.prep.discard_all_taken()
        regs.target.discard_all_taken()

    powered = np.linalg.matrix_power(_walk_matrix(), max(power, 0))
    expected = np.eye(8, dtype=complex)
    expected[4:, 4:] = powered
    assert_allclose_ignorephase(get_unitary(circuit, 3), expected)


@guppy
@no_type_check
def _generic_unitary_callback[n_prep: nat, n_target: nat](
    intrinsic: qubit,
    regs: QubitizationRegs[n_prep, array[qubit, n_target]],
    power: int,
    walk: Unitary[[qubit, array[qubit, n_prep], array[qubit, n_target]], None],
) -> None:
    qubitized_power_oracle(intrinsic, regs.prep, regs.target, power, walk)


@pytest.mark.parametrize(("n_prep", "n_target"), [(1, 1), (2, 3)])
def test_generic_unitary_callback_inference(n_prep: int, n_target: int) -> None:
    """Infer different PREPARE and target sizes through a generic Unitary argument."""

    @guppy
    @no_type_check
    def circuit(
        intrinsic: qubit,
        regs: QubitizationRegs[n_prep, array[qubit, n_target]],
    ) -> None:
        _generic_unitary_callback(intrinsic, regs, 2, _walk)

    circuit.compile_function()


@pytest.mark.parametrize("power", [0, 1, 2, 3])
def test_concrete_qubitized_walk_matrix(power: int) -> None:
    """Compare the production LCU/reflection walk to an independent RL matrix."""
    oracle = build_qubitized_power_oracle()

    @guppy
    @no_type_check
    def circuit(qs: array[qubit, 3]) -> None:
        regs = QubitizationRegs(array(qs.take(1)), array(qs.take(2)))
        oracle(qs[0], regs, power)
        qs.put(regs.prep.take(0), 1)
        qs.put(regs.target.take(0), 2)
        regs.prep.discard_all_taken()
        regs.target.discard_all_taken()

    hadamard = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    select = np.zeros((4, 4))
    select[:2, :2] = [[0, 1], [1, 0]]
    select[2:, 2:] = np.diag([1, -1])
    prepare = np.kron(hadamard, np.eye(2))
    reflection = np.diag([-1, -1, 1, 1])
    lcu = prepare @ select @ prepare
    assert not np.allclose(reflection @ lcu, lcu @ reflection)
    walk = reflection @ lcu
    expected = np.eye(8, dtype=complex)
    expected[4:, 4:] = np.linalg.matrix_power(walk, power)
    assert_allclose_ignorephase(get_unitary(circuit, 3), expected)


def test_runtime_powers_on_coherent_control() -> None:
    """Reuse one compiled program with runtime powers and a coherent control."""

    @guppy
    @no_type_check
    def main(power: int) -> None:
        qs = qarray(3)
        for i in range(3):
            h(qs[i])
        regs = QubitizationRegs(array(qs.take(1)), array(qs.take(2)))
        qubitized_power_oracle(qs[0], regs.prep, regs.target, power, _walk)
        qs.put(regs.prep.take(0), 1)
        qs.put(regs.target.take(0), 2)
        regs.prep.discard_all_taken()
        regs.target.discard_all_taken()
        state_output("result_state", qs)
        discard_array(qs)

    emulator = main.emulator(n_qubits=3)
    initial = np.ones(4) / np.sqrt(8)
    for power in (-1, 0, 1, 3):
        active = np.linalg.matrix_power(_walk_matrix(), max(power, 0)) @ initial
        # State snapshots use little endian; the reference is control/prep/target.
        expected = np.concatenate([initial, active]).reshape(2, 2, 2).T.flatten()
        actual = get_statevector(emulator, 3, main_args_dict={"power": power})
        assert_allclose_ignorephase(actual, expected)


def _make_modifier_application(mode: str, n_controls: int) -> GuppyFunctionDefinition:
    """Bind a modifier mode and external control count for the test programs.

    The returned Guppy signature is ``(qubit, array[qubit, 1], array[qubit, 1],
    int, Unitary[[qubit, array[qubit, 1], array[qubit, 1]], None],
    array[qubit, n_controls]) -> None``.
    """

    @guppy
    @no_type_check
    def apply(
        intrinsic: qubit,
        prep: array[qubit, 1],
        target: array[qubit, 1],
        runtime_power: int,
        walk: Unitary[[qubit, array[qubit, 1], array[qubit, 1]], None],
        external: array[qubit, n_controls],
    ) -> None:
        if comptime(mode == "forward"):
            qubitized_power_oracle(intrinsic, prep, target, runtime_power, walk)
        elif comptime(mode == "dagger"):
            with dagger:
                qubitized_power_oracle(intrinsic, prep, target, runtime_power, walk)
        elif comptime(mode == "control"):
            with control(external):
                qubitized_power_oracle(intrinsic, prep, target, runtime_power, walk)
        else:
            with control(external), dagger:
                qubitized_power_oracle(intrinsic, prep, target, runtime_power, walk)

    return apply


@pytest.mark.parametrize("power", [-1, 0, 1, 3])
@pytest.mark.parametrize("mode", ["forward", "dagger", "control", "controlled_dagger"])
@pytest.mark.parametrize("n_controls", [1, 2])
def test_power_oracle_modifiers(power: int, mode: str, n_controls: int) -> None:
    """Check explicit inverse loops and all control branches against full matrices."""
    n_qubits = n_controls + 3

    apply = _make_modifier_application(mode, n_controls)

    @guppy
    @no_type_check
    def circuit(qs: array[qubit, n_qubits]) -> None:
        external = array(qs.take(i) for i in range(n_controls))
        prep = array(qs.take(n_controls + 1))
        target = array(qs.take(n_controls + 2))
        apply(qs[n_controls], prep, target, power, _walk, external)
        for i in range(n_controls):
            qs.put(external.take(i), i)
        external.discard_all_taken()
        qs.put(prep.take(0), n_controls + 1)
        qs.put(target.take(0), n_controls + 2)
        prep.discard_all_taken()
        target.discard_all_taken()

    powered = np.linalg.matrix_power(_walk_matrix(), max(power, 0))
    if mode in ("dagger", "controlled_dagger"):
        powered = powered.conj().T
    intrinsic = np.eye(8, dtype=complex)
    intrinsic[4:, 4:] = powered
    if mode in ("control", "controlled_dagger"):
        expected = np.eye(2**n_qubits, dtype=complex)
        expected[-8:, -8:] = intrinsic
    else:
        expected = np.kron(np.eye(2**n_controls), intrinsic)
    assert_allclose_ignorephase(get_unitary(circuit, n_qubits), expected)


@pytest.mark.parametrize("mode", ["forward", "dagger", "control", "controlled_dagger"])
def test_runtime_modifiers_on_coherent_controls(mode: str) -> None:
    """Reuse one compiled modifier with runtime powers and coherent controls."""
    apply = _make_modifier_application(mode, 2)

    @guppy
    @no_type_check
    def main(power: int) -> None:
        external = qarray(2)
        intrinsic = qubit()
        prep = qarray(1)
        target = qarray(1)
        h(external[0])
        h(external[1])
        h(intrinsic)
        h(prep[0])
        h(target[0])
        apply(intrinsic, prep, target, power, _walk, external)
        qs = array(
            external.take(0), external.take(1), intrinsic, prep.take(0), target.take(0)
        )
        external.discard_all_taken()
        prep.discard_all_taken()
        target.discard_all_taken()
        state_output("result_state", qs)
        discard_array(qs)

    emulator = main.emulator(n_qubits=5)
    for power in (-1, 0, 1, 3):
        powered = np.linalg.matrix_power(_walk_matrix(), max(power, 0))
        if mode in ("dagger", "controlled_dagger"):
            powered = powered.conj().T
        expected = np.ones(32, dtype=complex) / np.sqrt(32)
        if mode in ("control", "controlled_dagger"):
            expected[-4:] = powered @ expected[-4:]
        else:
            for control_bits in range(4):
                start = control_bits * 8 + 4
                expected[start : start + 4] = powered @ expected[start : start + 4]
        expected = expected.reshape([2] * 5).T.flatten()
        actual = get_statevector(emulator, 5, main_args_dict={"power": power})
        assert_allclose_ignorephase(actual, expected)
