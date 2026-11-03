"""Public modifier integration checks for the migrated circuits."""

from typing import no_type_check

from guppylang import guppy
from guppylang.std.builtins import array, control, dagger
from guppylang.std.quantum import cx, h, qubit, t, tdg
from guppyalgos.testing import assert_unitary_modifiers, get_unitary

from guppyalgos.primitives.gate_decompositions.cnx.cnx import _rtof4, _tof5
from guppyalgos.primitives.gate_decompositions.cnx.cnx_cca import cat_cca
from guppyalgos.primitives.arithmetic.incrementer.incrementer_cca import (
    cat_cca as incrementer_cat,
)


def test_relative_toffoli_modifiers() -> None:
    """Check the optimized inverse and controls against a circuit reference."""

    @guppy
    @no_type_check
    def reference(qs: array[qubit, 4]) -> None:
        h(qs[3])
        t(qs[3])
        cx(qs[2], qs[3])
        tdg(qs[3])
        h(qs[3])
        cx(qs[0], qs[3])
        t(qs[3])
        cx(qs[1], qs[3])
        tdg(qs[3])
        cx(qs[0], qs[3])
        t(qs[3])
        cx(qs[1], qs[3])
        tdg(qs[3])
        h(qs[3])
        t(qs[3])
        cx(qs[2], qs[3])
        tdg(qs[3])
        h(qs[3])

    @guppy(unitary=True)
    @no_type_check
    def circuit(qs: array[qubit, 4]) -> None:
        _rtof4(qs[0], qs[1], qs[2], qs[3])

    expected = get_unitary(reference, 4)
    # The reference fixes |0000> with phase +1. Remove the simulator's overall
    # phase so controlled expectations preserve the phase between branches.
    expected /= expected[0, 0]
    assert_unitary_modifiers(circuit, 4, expected_unitary=expected)


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
