"""Public modifier integration checks for the migrated circuits."""

from typing import no_type_check

from guppylang import guppy
from guppylang.std.builtins import array, control, dagger
from guppylang.std.quantum import qubit
from guppyalgos.testing import assert_unitary_modifiers

from pytket.circuit import Circuit
from guppyalgos.primitives.gate_decompositions.cnx.cnx import _rtof4, _tof5
from guppyalgos.primitives.gate_decompositions.cnx.cnx_cca import cat_cca
from guppyalgos.primitives.arithmetic.incrementer.incrementer_cca import (
    cat_cca as incrementer_cat,
)


def test_relative_toffoli_modifiers() -> None:
    """Check the optimized inverse and controls against a circuit reference."""
    reference = Circuit(4)
    reference.H(3).T(3).CX(2, 3).Tdg(3).H(3)
    reference.CX(0, 3).T(3).CX(1, 3).Tdg(3)
    reference.CX(0, 3).T(3).CX(1, 3).Tdg(3)
    reference.H(3).T(3).CX(2, 3).Tdg(3).H(3)

    @guppy(unitary=True)
    @no_type_check
    def circuit(qs: array[qubit, 4]) -> None:
        _rtof4(qs[0], qs[1], qs[2], qs[3])

    assert_unitary_modifiers(circuit, 4, expected_unitary=reference.get_unitary())


def test_small_cnx_modifier_compilation() -> None:
    """Compile added controls and adjoints without imposing clean-workspace promises."""

    @guppy
    @no_type_check
    def circuit(qs: array[qubit, 6], cs: array[qubit, 2]) -> None:
        with control(cs), dagger:
            _tof5(qs[0], qs[1], qs[2], qs[3], qs[4], qs[5])
            cat_cca(qs[0], qs[1], qs[2])
            incrementer_cat(qs[0], qs[1], qs[2])

    circuit.compile_function()
