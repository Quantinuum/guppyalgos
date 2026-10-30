"""Single bit adder and half adders."""

from typing import no_type_check

from guppylang.std.quantum import toffoli, cx

from guppylang import guppy, qubit
from guppylang.std.builtins import dagger


@guppy
def full_adder(c_in: qubit, a: qubit, b: qubit, c_out: qubit) -> None:
    """Bitwise adder, sum bit computed onto b."""
    half_adder(c_in, a, c_out)
    half_adder(a, b, c_out)
    cx(c_in, a)


@guppy(unitary=True)
def half_adder(a: qubit, b: qubit, c_out: qubit) -> None:
    """Bitwise half-adder, sum bit computed onto b.

    Maps (a, b, c_out) to (a, a xor b, c_out xor (a and b)) on all inputs.
    Use ``with dagger`` for the inverse and ``with control`` for external controls.
    """
    toffoli(a, b, c_out)
    cx(a, b)


@guppy
@no_type_check
def full_adder_inverse(c_in: qubit, a: qubit, b: qubit, c_out: qubit) -> None:
    """Inverse of bitwise adder, sum bit computed onto b."""
    cx(c_in, a)
    with dagger:
        half_adder(a, b, c_out)
    with dagger:
        half_adder(c_in, a, c_out)
