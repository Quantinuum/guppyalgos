"""Tests for the BLISS-THC Select circuit skeleton."""

from random import Random
from typing import no_type_check

from guppylang import guppy
from guppylang.std.builtins import array
from guppylang.std.quantum import discard, discard_array, qubit

from guppyalgos.algorithms.block_encoding.thc import (
    SelectTHCCntrl,
    SelectTHCCntrlRegs,
    THCWalkTargetRegs,
)
from guppyalgos.algorithms.select.qrom import qrom_unary_iteration
from guppyalgos.primitives.gate_decompositions.cnx.cnx import cnx
from guppyalgos.primitives.rotations import (
    GivensCascadePhaseGradient,
)
from guppyalgos.utils import qarray

_N_INDEX_QUBITS = 3
_N_COMBINED_INDEX_QUBITS = _N_INDEX_QUBITS + 1
N_ORBS = 3
N_GIVENS = N_ORBS - 1
N_DATA_QUBITS = 4
_RNG = Random(42)
# The combined little-endian address is [mu..., c]. A three-qubit mu register
# therefore gives the combined QROM four address qubits and 16 rows.
_COMBINED_ANGLE_DATA = [
    [[bool(_RNG.getrandbits(1)) for _ in range(N_DATA_QUBITS)] for _ in range(N_GIVENS)]
    for _ in range(2 ** (_N_INDEX_QUBITS + 1))
]
_TWO_BODY_ANGLE_DATA = [
    [[bool(_RNG.getrandbits(1)) for _ in range(N_DATA_QUBITS)] for _ in range(N_GIVENS)]
    for _ in range(2**_N_INDEX_QUBITS)
]
_combined_qrom = qrom_unary_iteration(_COMBINED_ANGLE_DATA)
_two_body_qrom = qrom_unary_iteration(_TWO_BODY_ANGLE_DATA)


def test_thc_select_compiles_with_angle_loaders() -> None:
    """Check Select compiles with nontrivial combined-index and two-body QROMs."""

    @guppy
    @no_type_check
    def main() -> None:
        regs = SelectTHCCntrlRegs(
            qubit(),
            qubit(),
            qarray(_N_INDEX_QUBITS),
            qarray(_N_INDEX_QUBITS),
        )
        target_registers = THCWalkTargetRegs(
            qarray(N_ORBS),
            qarray(N_ORBS),
        )

        select = SelectTHCCntrl[
            _N_INDEX_QUBITS,
            _N_COMBINED_INDEX_QUBITS,
            N_DATA_QUBITS,
            N_GIVENS,
            N_ORBS,
            GivensCascadePhaseGradient[N_DATA_QUBITS, N_GIVENS, N_ORBS],
        ](
            _combined_qrom[array[array[qubit, N_DATA_QUBITS], N_GIVENS]],
            _two_body_qrom[array[array[qubit, N_DATA_QUBITS], N_GIVENS]],
            GivensCascadePhaseGradient[N_DATA_QUBITS, N_GIVENS, N_ORBS](
                qarray(N_DATA_QUBITS)
            ),
            cnx,
        )
        control = qubit()
        select.compose(control, regs, target_registers)

        discard(control)
        discard(regs.one_body_flag)
        discard(regs.coefficient_sign)
        discard_array(regs.first_index_qreg)
        discard_array(regs.second_index_qreg)
        discard_array(select.cascade.phase_gradient)
        discard_array(target_registers.spin_up)
        discard_array(target_registers.spin_down)

    main.check()
