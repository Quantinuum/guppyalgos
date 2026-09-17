"""Test hamming weight phasing."""

from typing import no_type_check

import pytest
from guppylang import guppy
from guppylang.std.angles import angle
from guppylang.std.debug import state_output
from guppylang.std.quantum import discard_array, h, rz
from selene_sim import Quest

from guppyalgos.primitives.rotations.hamming_weight_phasing import (
    hamming_weight_phase,
)
from guppyalgos.utils import qarray, transversal
from guppyalgos.testing import assert_allclose_ignorephase


@pytest.mark.parametrize(("n", "theta"), [(3, 0.1), (2, 0.5), (4, 0.25), (5, 1.0)])
def test_hamming_weight_phasing(n: int, theta: float) -> None:
    """Test hamming weight phasing produces the same state as parallel rotations."""
    hwp = hamming_weight_phase(n)

    @guppy
    @no_type_check
    def parallel_rotation() -> None:
        qs = qarray(n)
        transversal(h, qs)
        for i in range(len(qs)):
            rz(qs[i], angle(theta))
        state_output("out", qs)
        discard_array(qs)

    @guppy
    @no_type_check
    def hamming_weight_phased() -> None:
        qs = qarray(n)
        transversal(h, qs)
        hwp(qs, angle(theta))
        state_output("out", qs)
        discard_array(qs)

    res = parallel_rotation.emulator(n).run()
    par_state = Quest.extract_states_dict(res.results[0].entries)[
        "out"
    ].get_single_state()

    res = hamming_weight_phased.emulator(2 * n).run()
    hwp_state = Quest.extract_states_dict(res.results[0].entries)[
        "out"
    ].get_single_state()

    assert_allclose_ignorephase(par_state, hwp_state)
