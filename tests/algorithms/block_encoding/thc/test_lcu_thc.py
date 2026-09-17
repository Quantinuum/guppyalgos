"""Compile the nontrivial THC controlled LCU."""

from typing import no_type_check

import numpy as np
from guppylang import guppy
from guppylang.std.builtins import array, comptime
from guppylang.std.quantum import discard, discard_array, qubit

from guppyalgos.primitives.gate_decompositions.cnx.cnx import cnx
from guppyalgos.algorithms.block_encoding.lcu import LCUCntrl
from guppyalgos.primitives.rotations import GivensCascadePhaseGradient
from guppyalgos.algorithms.state_preparation.alias_sampling import (
    AliasSamplingRegs,
    alias_samp_prep,
    discard_alias_sampling_garbage,
)
from guppyalgos.primitives.state_preparation.phase_gradient import (
    Convention,
    phase_gradient,
)
from guppyalgos.algorithms.block_encoding.thc import (
    SelectTHCCntrl,
    SelectTHCCntrlRegs,
    THCParameters,
    THCWalkTargetRegs,
    build_thc_lcu_data,
    load_select_registers,
)
from guppyalgos.utils import qarray


N_INDEX_QUBITS = 3
N_ORBS = 3
N_GIVENS = N_ORBS - 1
N_DATA_QUBITS = 4
THC_RANK = 4
_RNG = np.random.default_rng(42)


def test_built_thc_cntrl_lcu_compiles() -> None:
    """Compile a nontrivial controlled LCU built from concrete THC data."""
    two_body_coefficients = _RNG.uniform(-1.0, 1.0, (THC_RANK, THC_RANK))
    two_body_coefficients = (two_body_coefficients + two_body_coefficients.T) / 2.0
    parameters = THCParameters(
        one_body_coefficients=_RNG.uniform(-1.0, 1.0, N_ORBS),
        two_body_coefficients=two_body_coefficients,
        one_body_rotations=_RNG.random((N_ORBS, N_GIVENS)),
        two_body_rotations=_RNG.random((THC_RANK, N_GIVENS)),
    )
    thc_lcu_data = build_thc_lcu_data(
        parameters,
        rotation_precision_bits=N_DATA_QUBITS,
        alias_precision_bits=4,
    )
    assert thc_lcu_data.n_index_qubits == N_INDEX_QUBITS
    assert thc_lcu_data.n_modes == N_ORBS
    assert thc_lcu_data.n_givens == N_GIVENS
    alias_prepare = alias_samp_prep(
        thc_lcu_data.alias_probabilities,
        thc_lcu_data.alias_precision,
    )
    select_qrom = thc_lcu_data.select_data_loader
    qrom_1_and_2_body = thc_lcu_data.qrom_1_and_2_body
    qrom_2_body = thc_lcu_data.qrom_2_body
    n_alias_qubits = thc_lcu_data.n_alias_qubits
    n_keep_qubits = thc_lcu_data.n_keep_qubits
    n_modes = thc_lcu_data.n_modes
    n_phase_qubits = thc_lcu_data.rotation_precision_bits
    prepare_phase_gradient = phase_gradient(
        n_phase_qubits,
        convention=Convention.Standard,
    )

    @guppy.struct
    class THCPrepareRegs:
        """Registers shared by the three THC LCU oracles."""

        alias_sampling: AliasSamplingRegs[n_alias_qubits, n_keep_qubits]
        select: SelectTHCCntrlRegs[N_INDEX_QUBITS]
        phase_gradient: array[qubit, n_phase_qubits]

    @guppy
    @no_type_check
    def prepare(regs: THCPrepareRegs) -> None:
        alias_prepare(
            regs.alias_sampling.index,
            regs.alias_sampling.alternative,
            regs.alias_sampling.keep,
            regs.alias_sampling.comparison,
            regs.alias_sampling.comparison_result,
            False,
        )
        load_select_registers(
            select_qrom[array[qubit, comptime(2 * N_INDEX_QUBITS + 2)]],
            regs.alias_sampling.index,
            regs.select,
        )

    @guppy
    @no_type_check
    def cntrl_select(
        control: qubit,
        prep_regs: THCPrepareRegs,
        target_regs: THCWalkTargetRegs[n_modes],
    ) -> None:
        cascade = GivensCascadePhaseGradient[n_phase_qubits, N_GIVENS, n_modes](
            prep_regs.phase_gradient
        )
        select = SelectTHCCntrl(
            qrom_1_and_2_body[array[array[qubit, n_phase_qubits], N_GIVENS]],
            qrom_2_body,
            cascade,
            cnx,
        )
        select.compose(control, prep_regs.select, target_regs)
        prep_regs.phase_gradient = select.cascade.phase_gradient

    @guppy
    @no_type_check
    def unprepare(regs: THCPrepareRegs) -> None:
        load_select_registers(
            select_qrom[array[qubit, comptime(2 * N_INDEX_QUBITS + 2)]],
            regs.alias_sampling.index,
            regs.select,
        )
        alias_prepare(
            regs.alias_sampling.index,
            regs.alias_sampling.alternative,
            regs.alias_sampling.keep,
            regs.alias_sampling.comparison,
            regs.alias_sampling.comparison_result,
            True,
        )

    @guppy
    @no_type_check
    def main() -> None:
        control = qubit()
        alias_regs = AliasSamplingRegs(
            qarray(n_alias_qubits),
            qarray(n_alias_qubits),
            qarray(n_keep_qubits),
            qarray(n_keep_qubits),
            qubit(),
        )
        select_regs = SelectTHCCntrlRegs(
            qubit(),
            qubit(),
            qarray(N_INDEX_QUBITS),
            qarray(N_INDEX_QUBITS),
        )
        target_regs = THCWalkTargetRegs(
            qarray(n_modes),
            qarray(n_modes),
        )
        phase_gradient = qarray(n_phase_qubits)
        prepare_phase_gradient(phase_gradient)
        cntrl_lcu = LCUCntrl(
            prepare,
            cntrl_select,
            unprepare,
        )
        prep_regs = THCPrepareRegs(alias_regs, select_regs, phase_gradient)

        cntrl_lcu.compose(control, prep_regs, target_regs)

        discard(control)
        index = discard_alias_sampling_garbage(
            prep_regs.alias_sampling.index,
            prep_regs.alias_sampling.alternative,
            prep_regs.alias_sampling.keep,
            prep_regs.alias_sampling.comparison,
            prep_regs.alias_sampling.comparison_result,
        )
        discard_array(index)
        discard(prep_regs.select.one_body_flag)
        discard(prep_regs.select.coefficient_sign)
        discard_array(prep_regs.select.first_index_qreg)
        discard_array(prep_regs.select.second_index_qreg)
        discard_array(target_regs.spin_up)
        discard_array(target_regs.spin_down)
        discard_array(prep_regs.phase_gradient)

    main.compile()
