"""Test hamming weight calculation."""

from typing import no_type_check

import numpy as np
import pytest
from guppylang import guppy
from guppylang.std.builtins import output
from guppylang.std.debug import state_output
from guppylang.std.quantum import (
    collect_measurements,
    discard_array,
    h,
    measure_array,
    x,
)
from selene_sim import Quest

from guppyalgos.primitives.arithmetic.hamming_weight import (
    hamming_weight_func,
    hamming_weight_func_inv,
    num_hamming_weight_bits,
)
from guppyalgos.utils import int_to_bits, qarray, transversal
from guppyalgos.testing import assert_allclose_ignorephase, switch_endianness


@pytest.mark.parametrize(
    ("n", "init_bits"),
    [
        (3, [True, True, False]),
        (3, [True, False, True]),
        (4, [True, True, False, False]),
        (4, [True, True, True, True]),
        (4, [True, False, True, False]),
        (5, [True, True, False, False, False]),
        (5, [True, False, True, False, False]),
        (6, [True, True, False, False, False, False]),
        (6, [True, False, True, False, False, False]),
    ],
)
def test_hamming_weight(n: int, init_bits: list[bool]):
    """Checks that the hamming weight computed is as expected."""
    ham = hamming_weight_func(n)
    n_ham_bits = num_hamming_weight_bits(n)
    expected_hamming_weight = sum(init_bits)

    @guppy
    @no_type_check
    def main() -> None:
        """Compute hamming weight for a bitstring input."""
        m = qarray(n)
        for i in range(n):
            if init_bits[i]:
                x(m[i])
        h, j = ham(m)
        m.discard_all_taken()
        state_output("hamm", h)
        output("meas", collect_measurements(measure_array(h)))
        discard_array(j)

    res = main.emulator(2 * n).run()
    assert res.results[0].as_dict()["meas"] == int_to_bits(
        expected_hamming_weight, n_ham_bits
    )
    states = Quest.extract_states_dict(res.results[0].entries)
    state = states["hamm"].get_single_state()

    # read out the hamming weight from the statevector
    non_zero_index = np.nonzero(switch_endianness(state))
    # assert we have a pure c-basis state
    assert len(non_zero_index) == 1
    non_zero_index = non_zero_index[0][0]
    ham_reg_bits = int_to_bits(non_zero_index, n_ham_bits)
    hamming_weight_calculated = sum(bit * 2**i for i, bit in enumerate(ham_reg_bits))
    assert hamming_weight_calculated == expected_hamming_weight


@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_hamming_weight_inverse(n: int) -> None:
    """Checks that ham_inv acts as inverse for uniform superposition."""
    ham = hamming_weight_func(n)
    ham_inv = hamming_weight_func_inv(n)

    @guppy
    @no_type_check
    def main() -> None:
        """Comp and uncomp hamming weight on uniform state."""
        main_reg = qarray(n)
        transversal(h, main_reg)
        ham_reg, junk_reg = ham(main_reg)
        ham_inv(main_reg, ham_reg, junk_reg)
        transversal(h, main_reg)
        state_output("main", main_reg)
        discard_array(main_reg)

    res = main.emulator(2 * n).run()
    states = Quest.extract_states_dict(res.results[0].entries)
    zero_state = np.zeros(2**n)
    zero_state[0] = 1
    main_state = states["main"].get_single_state()
    assert_allclose_ignorephase(main_state, zero_state)
