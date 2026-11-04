"""Tests for the CCX V chains."""

import random
from typing import no_type_check

import pytest
from guppylang import guppy
from guppylang.std.builtins import array, output
from guppylang.std.quantum import (
    collect_measurements,
    measure,
    measure_array,
    qubit,
    x,
)

from guppyalgos.primitives.subroutines.ladders.ccx_v_chain import (
    ccx_v_chain,
    ccx_v_chain_logdepth,
)
from guppyalgos.primitives.utilities.gates import apply_bitstring
from guppyalgos.utils import qarray
from guppyalgos.utils import int_to_bits


@pytest.mark.parametrize("implementation", ["linear", "logdepth"])
@pytest.mark.parametrize("n", [1, 2, 6, 8, 10])
def test_ccx_v_chain(implementation: str, n: int) -> None:
    """Test ccx v chain."""
    if implementation == "linear":
        chain_fn = ccx_v_chain
    else:
        chain_fn = ccx_v_chain_logdepth(n)

    @guppy
    @no_type_check
    def main(a_bits: array[bool, n], b_bits: array[bool, n], flip: int) -> None:
        """Prepare basis inputs, apply the V chain, and measure."""
        controls_a = qarray(n)
        controls_b = qarray(n)
        target = qubit()
        apply_bitstring(controls_a, a_bits)
        apply_bitstring(controls_b, b_bits)
        if flip == 1:
            x(target)
        chain_fn(controls_a, controls_b, target)
        output("controls_a", collect_measurements(measure_array(controls_a)))
        output("controls_b", collect_measurements(measure_array(controls_b)))
        output("target", measure(target).read())

    emulator = main.emulator(n_qubits=2 * n + 1).with_seed(42).with_shots(1)

    rng = random.Random(n)
    cases = [
        (rng.randrange(2**n), rng.randrange(2**n), rng.randrange(2)) for _ in range(10)
    ]

    for a_value, b_value, flip in cases:
        a_bits, b_bits = int_to_bits(a_value, n), int_to_bits(b_value, n)
        shot = emulator.run(a_bits=a_bits, b_bits=b_bits, flip=flip).collated_shots()[0]

        chain = a_bits[0]
        for i in range(1, n):
            chain = a_bits[i] ^ (b_bits[i - 1] and chain)
        expected_target = bool(flip) ^ (chain and b_bits[-1])

        case = (a_value, b_value, flip)
        assert shot["controls_a"][0] == a_bits, case
        assert shot["controls_b"][0] == b_bits, case
        assert shot["target"][0] == expected_target, case
