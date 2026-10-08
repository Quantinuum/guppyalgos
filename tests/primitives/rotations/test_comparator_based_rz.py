"""Tests for single-qubit rotation algorithm."""

import numpy as np
import pytest
from math import ceil, log2, pi
from typing import no_type_check

from guppylang import guppy, comptime
from guppylang.std.builtins import nothing, output, some
from guppylang.std.debug import state_output
from guppylang.std.angles import angle
from guppylang.std.quantum import qubit, h, discard, toffoli
from selene_sim import QuantumReplay, Quest

from guppyalgos.primitives.rotations import (
    ComparatorBasedRz,
    ConstantComparatorCascade,
    comparator_based_rz_cascade,
    n_comparator_based_rz_cascade_ancillas,
    n_constant_comparator_cascade_ancillas,
)


@pytest.mark.parametrize("epsilon", [0.1, 0.01, 0.001])
def test_default_ancilla_count(epsilon: float) -> None:
    """Count the RUS register and default comparator workspace."""
    n = 1 + ceil(log2(1 / epsilon))

    assert n_comparator_based_rz_cascade_ancillas(epsilon) == 2 * n - 2


@pytest.mark.parametrize(
    ("epsilon", "theta"),
    [
        (0.1, pi / 4),
        (0.01, 0.5),
        (0.01, 0.0),
        (0.01, pi / 2),
        (0.01, -3 * pi / 4),
    ],
)
def test_comparator_based_rz(
    epsilon: float,
    theta: float,
) -> None:
    """Test that Rz(theta) gate is approximated within error bound epsilon."""
    rz_fn = comparator_based_rz_cascade(epsilon)

    theta = theta / pi

    @guppy
    @no_type_check
    def test_circuit() -> None:
        target = qubit()
        h(target)
        rz_fn(target, angle(comptime(theta)))
        state_output("final", target)
        discard(target)

    result = test_circuit.emulator(
        n_comparator_based_rz_cascade_ancillas(epsilon) + 1
    ).run()
    states = Quest.extract_states_dict(result.results[0].entries)
    final_state = states["final"].get_single_state()
    theta_star = np.angle(final_state[1] / final_state[0]) / np.pi
    angle_error = abs(np.angle(np.exp(1j * np.pi * (theta - theta_star))))

    assert angle_error < epsilon, (
        f"Angle error {angle_error:.6e} exceeds epsilon {epsilon}"
    )


@pytest.mark.parametrize("theta", [0.1, 0.3, 0.5, 0.9])
def test_comparator_based_rz_replay(theta: float) -> None:
    """Test the repeat-until-success while loop via replay simulation."""
    epsilon = 0.01
    n = 1 + ceil(log2(1 / epsilon))
    n_comparator_ancillas = n_constant_comparator_cascade_ancillas(n)

    @guppy
    @no_type_check
    def rz_fn(target: qubit, theta: angle) -> None:
        comparator = ConstantComparatorCascade[
            comptime(n), comptime(n_comparator_ancillas)
        ](toffoli, toffoli, False)
        inverse_comparator = ConstantComparatorCascade[
            comptime(n), comptime(n_comparator_ancillas)
        ](toffoli, toffoli, True)
        rz = ComparatorBasedRz(
            comparator,
            inverse_comparator,
            nothing(),
        )
        rz.compose(target, theta)

    theta = theta / pi  # scale theta to [0, pi]

    @guppy
    @no_type_check
    def circ_rus() -> None:
        target = qubit()
        h(target)
        rz_fn(target, angle(comptime(theta)))
        state_output("final", target)
        discard(target)

    n_shots = 5
    # attempts fail k times and then succeed
    fail = [False] * (2 * n - 3) + [True]
    desired_measurements = [fail * k + [False] * (2 * n - 2) for k in range(n_shots)]

    rus_replay_sim = QuantumReplay(simulator=Quest(), measurements=desired_measurements)
    em_result = (
        circ_rus.emulator(n_comparator_based_rz_cascade_ancillas(epsilon) + 1)
        .with_simulator(rus_replay_sim)
        .with_shots(n_shots)
        .run()
    )
    for shot_result in em_result.results:
        states = Quest.extract_states_dict(shot_result.entries)
        final_state = states["final"].get_single_state()
        theta_star = np.angle(final_state[1] / final_state[0]) / np.pi
        angle_error = abs(np.angle(np.exp(1j * np.pi * (theta - theta_star))))
        assert angle_error < epsilon, (
            f"Angle error {angle_error:.6e} exceeds epsilon {epsilon}"
        )


def test_comparator_based_rz_max_attempts() -> None:
    """Test that a shot exits once ``max_attempts`` attempts have failed."""
    epsilon = 0.01
    max_attempts = 3
    n = 1 + ceil(log2(1 / epsilon))
    n_comparator_ancillas = n_constant_comparator_cascade_ancillas(n)

    @guppy
    @no_type_check
    def circ_rus() -> None:
        comparator = ConstantComparatorCascade[
            comptime(n), comptime(n_comparator_ancillas)
        ](toffoli, toffoli, False)
        inverse_comparator = ConstantComparatorCascade[
            comptime(n), comptime(n_comparator_ancillas)
        ](toffoli, toffoli, True)
        rz = ComparatorBasedRz(
            comparator,
            inverse_comparator,
            some(comptime(max_attempts)),
        )
        target = qubit()
        h(target)
        rz.compose(target, angle(0.1))
        output("done", True)
        discard(target)

    fail = [False] * (2 * n - 3) + [True]
    success = [False] * (2 * n - 2)
    desired_measurements = [
        fail * (max_attempts - 1) + success,
        fail * max_attempts,
        success,
    ]

    rus_replay_sim = QuantumReplay(simulator=Quest(), measurements=desired_measurements)
    em_result = (
        circ_rus.emulator(n_comparator_based_rz_cascade_ancillas(epsilon) + 1)
        .with_simulator(rus_replay_sim)
        .with_shots(len(desired_measurements))
        .run()
    )

    assert [shot.entries for shot in em_result.results] == [
        [("attempts", max_attempts), ("done", 1)],
        [("exit: ComparatorBasedRz reached max_attempts without success", 1)],
        [("attempts", 1), ("done", 1)],
    ]
