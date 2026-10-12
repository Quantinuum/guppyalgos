"""Tests for the ladders of multi-controlled X gates."""

import random
from typing import no_type_check

import pytest
from guppylang import guppy
from guppylang.std.builtins import array, output
from guppylang.std.quantum import (
    collect_measurements,
    discard,
    measure,
    measure_array,
    qubit,
    x,
)

from guppyalgos.primitives.subroutines.ladders import (
    cnx_ladder_logdepth,
    cnx_ladder_logdepth_num_ancilla,
)
from guppyalgos.utils import apply_bitstring, int_to_bits, qarray


@pytest.mark.parametrize(("n", "k"), [(3, 3), (4, 2)])
def test_cnx_ladder(n: int, k: int) -> None:
    """Test cnx ladder."""
    n_controls_b = k - 1
    n_b = n * n_controls_b
    n_dirty = 2 * n
    n_ancillae = cnx_ladder_logdepth_num_ancilla(n)
    ladder_fn = cnx_ladder_logdepth(n, k)

    @guppy
    @no_type_check
    def main(
        a_bits: array[bool, n],
        b_bits: array[bool, n_b],
        dirty_bits: array[bool, n_dirty],
        flip: int,
    ) -> None:
        """Prepare basis inputs, apply the ladder, and measure."""
        controls_a = qarray(n)
        controls_b = array(qarray(n_controls_b) for _ in range(n))
        target = qubit()
        borrowed_a = qarray(n)
        borrowed_b = qarray(n)
        ancillae = qarray(n_ancillae)
        apply_bitstring(controls_a, a_bits)
        for i in range(n):
            for j in range(n_controls_b):
                if b_bits[i * n_controls_b + j]:
                    x(controls_b[i][j])
            if dirty_bits[i]:
                x(borrowed_a[i])
            if dirty_bits[n + i]:
                x(borrowed_b[i])
        if flip == 1:
            x(target)
        ladder_fn(controls_a, controls_b, target, borrowed_a, borrowed_b, ancillae)
        for ancilla in ancillae:
            discard(ancilla)
        output("controls_a", collect_measurements(measure_array(controls_a)))
        for register in controls_b:
            output("controls_b", collect_measurements(measure_array(register)))
        output("borrowed_a", collect_measurements(measure_array(borrowed_a)))
        output("borrowed_b", collect_measurements(measure_array(borrowed_b)))
        output("target", measure(target).read())

    n_qubits = n * k + 1 + n_dirty + n_ancillae
    emulator = main.emulator(n_qubits=n_qubits).with_seed(42).with_shots(1)

    rng = random.Random(10 * n + k)
    cases = [
        (
            rng.randrange(2**n),
            rng.randrange(2**n_b),
            rng.randrange(2**n_dirty),
            rng.randrange(2),
        )
        for _ in range(10)
    ]

    for a_value, b_value, dirty_value, flip in cases:
        a_bits, b_bits = int_to_bits(a_value, n), int_to_bits(b_value, n_b)
        dirty_bits = int_to_bits(dirty_value, n_dirty)
        shot = emulator.run(
            a_bits=a_bits, b_bits=b_bits, dirty_bits=dirty_bits, flip=flip
        ).collated_shots()[0]

        b_groups = [b_bits[i * n_controls_b : (i + 1) * n_controls_b] for i in range(n)]
        expected_a = list(a_bits)
        for i in range(n - 1):
            expected_a[i + 1] ^= expected_a[i] and all(b_groups[i])
        expected_target = bool(flip) ^ (expected_a[-1] and all(b_groups[-1]))

        case = (a_value, b_value, dirty_value, flip)
        assert shot["controls_a"][0] == expected_a, case
        assert shot["controls_b"] == b_groups, case
        assert shot["borrowed_a"][0] == dirty_bits[:n], case
        assert shot["borrowed_b"][0] == dirty_bits[n:], case
        assert shot["target"][0] == expected_target, case
