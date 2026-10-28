"""Perform parallel z-rotations based on hamming weight phasing."""

from typing import no_type_check

from guppylang.std.builtins import nat

from guppylang import array, guppy, qubit
from guppylang.std.angles import angle
from guppylang.std.quantum import rz

from guppyalgos.primitives.arithmetic.hamming_weight import (
    hamming_weight,
    hamming_weight_uncompute,
)


@guppy
@no_type_check
def hamming_weight_phase[n: nat](qs: array[qubit, n], angle: angle) -> None:
    """Perform a layer of identical rotations via hamming weight phasing.

    Hamming weight phasing performs a layer of parallel rz gates with the same angle
    on a register using the observation that the total phase applied depends only on
    the hamming weight of the state.

    The hamming weight is calculated using a log-depth sequence of adders,
    log2(n) rotations are applied on the hamming weight register, and then the
    hamming weight is uncomputed.
    """
    ham_reg, junk = hamming_weight(qs)
    for i in range(len(ham_reg)):
        rz(ham_reg[i], 2**i * angle)
    hamming_weight_uncompute(qs, ham_reg, junk)
