"""Single bit adder and half adders."""

from guppylang.std.quantum import toffoli, cx

from guppylang import guppy, qubit


@guppy(unitary=True)
def full_adder(c_in: qubit, a: qubit, b: qubit, c_out: qubit) -> None:
    """Bitwise adder, sum bit computed onto b."""
    half_adder(c_in, a, c_out)
    half_adder(a, b, c_out)
    cx(c_in, a)


@guppy(unitary=True)
def half_adder(a: qubit, b: qubit, c_out: qubit) -> None:
    """Bitwise half-adder, sum bit computed onto b."""
    toffoli(a, b, c_out)
    cx(a, b)
