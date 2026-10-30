"""Exponentiator using Gidney's ripple-carry adder."""

from typing import no_type_check

from guppylang import guppy
from guppylang.std.builtins import array, comptime, nat
from guppylang.std.quantum import qubit, discard_array

from guppyalgos.primitives.arithmetic.multiplier.multiplier_ripple_gidney import (
    cntrl_multiplier_ripple_gidney_mod_in_place,
)
from guppyalgos.utils.guppy.unsafe_borrow import (
    _unsafe_array_borrow,
    _unsafe_array_borrow_slice,
    _unsafe_array_unborrow,
    _unsafe_array_unborrow_slice,
)
from guppyalgos.utils import qarray
from guppyalgos.utils.guppy.array import join_arrays, split_array


@guppy.comptime
@no_type_check
def _require_valid_base(base: int @ comptime) -> None:
    """Require a non-negative odd base for reversible exponentiation."""
    if base < 0:
        msg = "In-place exponentiation requires a non-negative base"
        raise ValueError(msg)
    if base % 2 == 0:
        msg = "In-place exponentiation modulo 2**n requires an odd base"
        raise ValueError(msg)


@guppy
@no_type_check
def exponentiator_ripple_gidney_mod[n_exponent: nat, n_output: nat](
    exponent_reg: array[qubit, n_exponent],
    output_reg: array[qubit, n_output],
    base: int @ comptime,
) -> None:
    """Multiply the output register by ``base^exponent`` modulo ``2**n_output``.

    Base must be odd since the multiplication is performed in-place, and multiplication
    modulo ``2**n_output`` requires an odd multiplicand for reversibility.

    Explicitly, performs the transformation
    ``|x⟩|a⟩ → |x⟩|a * base^x mod 2**n_output⟩``.

    Repeated squaring uses static recursive calls so Guppy can resolve the
    underlying adders' custom modifiers. An empty output is a no-op; padding
    keeps its unreachable multiplier specialization well typed. Invalid bases
    are rejected before execution, with dependent specializations kept odd.

    Args:
        exponent_reg: Quantum register holding the exponent.
        output_reg: Quantum register to store the result. Typically initialized to
            ``|1⟩``.
        base: Classical base to be exponentiated.

    Raises:
        ValueError: If the base is negative or even.

    """
    _require_valid_base(base)
    if n_exponent == 0 or n_output == 0:
        return
    padded_output = join_arrays(
        _unsafe_array_borrow(output_reg),
        qarray(comptime(1 if n_output == 0 else 0)),
        comptime(max(1, n_output)),
    )
    cntrl_multiplier_ripple_gidney_mod_in_place(
        exponent_reg[0], padded_output, comptime((base % 2 ** max(1, n_output)) | 1)
    )
    restored_output, padding = split_array(
        padded_output, n_output, comptime(1 if n_output == 0 else 0)
    )
    _unsafe_array_unborrow(output_reg, restored_output)
    discard_array(padding)
    prefix, remaining, suffix = _unsafe_array_borrow_slice(
        exponent_reg, comptime(min(1, n_exponent)), comptime(max(0, n_exponent - 1)), 0
    )
    exponentiator_ripple_gidney_mod(
        remaining, output_reg, comptime((base * base % 2 ** max(1, n_output)) | 1)
    )
    _unsafe_array_unborrow_slice(exponent_reg, prefix, remaining, suffix)
