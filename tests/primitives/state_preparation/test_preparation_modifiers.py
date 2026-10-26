"""Public modifier integration checks for the migrated circuits."""

from typing import no_type_check

import numpy as np
from guppylang import guppy
from guppylang.std.builtins import array, control, dagger
from guppylang.std.quantum import qubit

from guppyalgos.primitives.state_preparation.phase_gradient import phase_gradient
from guppyalgos.algorithms.state_preparation.multiplexor_prep import multiplexor_prep
from guppyalgos.algorithms.state_preparation.diagonal.walsh import (
    diagonal_unitary_walsh,
)


def test_preparation_modifier_compilation() -> None:
    """Compile generated controls and adjoints on the entire circuit domain."""
    gradient = phase_gradient(2)
    prepare = multiplexor_prep(np.array([1, 2j, -3, 4j]) / np.sqrt(30))
    diagonal = diagonal_unitary_walsh(np.exp(1j * np.array([0.1, -0.2, 0.3, -0.4])))

    @guppy
    @no_type_check
    def circuit(qs: array[qubit, 2], cs: array[qubit, 2]) -> None:
        with control(cs), dagger:
            gradient(qs)
            prepare(qs)
            diagonal(qs)

    circuit.compile_function()
