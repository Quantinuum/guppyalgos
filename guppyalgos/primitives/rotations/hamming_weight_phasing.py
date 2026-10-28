"""Perform parallel z-rotations based on hamming weight phasing."""

from typing import no_type_check

from guppylang.std.builtins import nat

from guppylang import array, guppy, qubit
from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.angles import angle
from guppylang.std.quantum import rz

from guppyalgos.primitives.arithmetic.hamming_weight import (
    hamming_weight,
    hamming_weight_func_inv,
)


def hamming_weight_phase[n_q: nat](
    n: int, rz_method: GuppyFunctionDefinition[[qubit, angle], None] = rz
) -> GuppyFunctionDefinition[[array[qubit, n_q], angle], None]:
    """Produce a guppy function to perform hamming weight phasing.

    Hamming weight phasing performs a layer of parallel rz gates with the same angle
    on a register using the observation that the total phase applied depends only on the
    hamming weight of the state.

    The hamming weight is calculated using a log-depth sequence of adders,
    log2(n) rotations are applied on the hamming weight register, and then the
    hamming weight is uncomputed.
    """
    ham_uncompute = hamming_weight_func_inv(n)

    @guppy
    @no_type_check
    def hwp(qs: array[qubit, n], angle: angle) -> None:
        ham_reg, junk = hamming_weight(qs)
        for i in range(len(ham_reg)):
            rz_method(ham_reg[i], 2**i * angle)
        ham_uncompute(qs, ham_reg, junk)

    return hwp
