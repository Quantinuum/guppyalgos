r"""Alias sampling state preparation for probability distributions.

This module implements the alias sampling state preparation algorithm,
given :math:`\rho_l` as a probability distribution over a finite set of outcomes
:math:`l \in \{1, \dots, L\}`, the circuit outputs the state

.. math::

    |\psi\rangle = \sum_{l=0}^{L-1}\sqrt{\tilde{\rho}_l}
    |l\rangle|\mathrm{temp}_l\rangle,

where :math:`\tilde{\rho}_l` approximates :math:`\rho_l`, and
:math:`|\mathrm{temp}_l\rangle` denotes the ancilla registers used in the alias
sampling process.

Ref:
    Babbush, R., Gidney, C., Berry, D. W., Wiebe, N., McClean, J., Paler, A.,
    ... & Neven, H. (2018). Encoding electronic spectra in quantum circuits
    with linear T complexity. Physical Review X, 8(4), 041015.
"""

from guppyalgos.primitives.subroutines.fanout import fanout_basic, fanout_from_data
from guppyalgos.utils.guppy.array import join_arrays, split_array

from math import ceil, log2
from typing import Protocol, no_type_check

import numpy as np
import numpy.typing as npt
from guppylang import guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.builtins import (
    array,
    comptime,
    nat,
    owned,
)
from guppylang.std.quantum import discard, discard_array, qubit, x

from guppyalgos.primitives.arithmetic.comparator import (
    comparator_ripple_cuccaro,
)
from guppyalgos.algorithms.select.qrom import qrom_unary_iteration
from guppyalgos.primitives.state_preparation import uniform_state
from guppyalgos.algorithms.state_preparation.alias_sampling.alias_table import Aliaser
from guppyalgos.utils import (
    _unsafe_array_borrow,
    _unsafe_array_unborrow,
    cswap,
    float_to_fixed_point,
    int_to_bits,
)


class UniformWithDagger(Protocol):
    """Protocol for specifying that uniform_prep has a dagger kwarg."""

    def __call__(self, n_amps: int, *, dagger: bool = False) -> GuppyFunctionDefinition:
        """Generate uniform state prep function."""
        ...


@guppy.struct
class AliasSamplingRegs[n_index_q: nat, n_keep_q: nat]:
    """Registers retained between alias-sampling PREPARE and UNPREPARE.

    Args:
        index: Index selecting an outcome in the prepared distribution.
        alternative: Workspace holding the alternative alias-table index.
        keep: Workspace holding the alias-table acceptance threshold.
        comparison: Uniform comparison workspace.
        comparison_result: Qubit holding the comparison result.

    """

    index: array[qubit, n_index_q]
    alternative: array[qubit, n_index_q]
    keep: array[qubit, n_keep_q]
    comparison: array[qubit, n_keep_q]
    comparison_result: qubit


def alias_samp_prep[n: nat, n_prob_bits: nat](
    prob_dist: npt.NDArray[np.float64],
    precision: float,
    uniform_state_prep_box: UniformWithDagger = uniform_state,
    compare_box: GuppyFunctionDefinition[
        [
            array[qubit, n_prob_bits],
            array[qubit, n_prob_bits],
            qubit,
        ],
        None,
    ] = comparator_ripple_cuccaro,
    fanout_op: GuppyFunctionDefinition = fanout_basic,
) -> GuppyFunctionDefinition[
    [
        array[qubit, n],
        array[qubit, n],
        array[qubit, n_prob_bits],
        array[qubit, n_prob_bits],
        qubit,
        bool,
    ],
    None,
]:
    r"""Alias sampling state preparation.

    Given :math:`\rho_l` as a probability distribution over a finite set of outcomes
    :math:`l \in \{1, \dots, L\}`, the circuit outputs the state

    .. math::

        |\psi\rangle = \sum_{l=0}^{L-1}\sqrt{\tilde{\rho}_l}
        |l\rangle|\mathrm{temp}_l\rangle,

    where :math:`\tilde{\rho}_l` approximates :math:`\rho_l`, and
    :math:`|\mathrm{temp}_l\rangle` denotes the ancillary registers used in the alias
    sampling process.

    Args:
        prob_dist (NDArray[float]): probability distribution to be loaded.
        precision (float): Required precision for the loaded proababilities.
            (to the nearest fixed point binary number).
        uniform_state_prep_box: Guppy function for uniform state preparation.
        compare_box: Guppy function for comparison operation.
        fanout_op: Flat fanout operation applied inside the QROM.

    Returns:
        Guppy function which implements the alias sampling preparation, see alias_box
        below for args

    """
    n_index_qubits = ceil(log2(len(prob_dist)))
    n_keep_prob_qubits = ceil(log2(1 / precision))
    alias_table = Aliaser(prob_dist, n_keep_prob_qubits)
    qrom_input = qrom_bitstrings_from_alias_table(alias_table, precision)

    alias_qrom = qrom_unary_iteration(
        qrom_input,
        fanout_op=fanout_op,
        fanout_from_data_fn=build_alias_fanout,
    )
    input_index_prep = uniform_state_prep_box(len(prob_dist), dagger=False)
    input_index_prep_dagger = uniform_state_prep_box(len(prob_dist), dagger=True)
    # self inverse for 2**n
    comparison_prob_prep = uniform_state_prep_box(2**n_keep_prob_qubits)

    @guppy
    @no_type_check
    def alias_box(
        index_reg: array[qubit, n_index_qubits],
        alternative_val_reg: array[qubit, n_index_qubits],
        keep_prob_reg: array[qubit, n_keep_prob_qubits],
        compare_alternative_val_reg: array[qubit, n_keep_prob_qubits],
        compare_out: qubit,
        dagger: bool,
    ) -> None:
        """Alias sampling state preparation.

        Args:
            index_reg (array[qubit]): Index register for the output probability
                distribution.
            alternative_val_reg (array[qubit]): Junk register used for loading
                alternative index values.
            keep_prob_reg (array[qubit]): Junk register used internally for
                loading probabilities to choose between original and alternative
                values.
            compare_alternative_val_reg (array[qubit]):
                Junk register storing a uniform distribution for comparison with
                keep_prob.
            compare_out (qubit):
                Result of comparison bit.
            dagger (bool):
                Apply the inverse operation if True

        """
        if not dagger:
            input_index_prep(
                index_reg,
            )

            alternative_val_reg_temp = _unsafe_array_borrow(alternative_val_reg)
            keep_prob_reg_temp = _unsafe_array_borrow(keep_prob_reg)
            alt_and_keep = (alternative_val_reg_temp, keep_prob_reg_temp)

            alias_qrom(index_reg, alt_and_keep)

            _unsafe_array_unborrow(alternative_val_reg, alt_and_keep[0])
            _unsafe_array_unborrow(keep_prob_reg, alt_and_keep[1])

            comparison_prob_prep(compare_alternative_val_reg)
            compare_box(compare_alternative_val_reg, keep_prob_reg, compare_out)
            x(compare_out)
            cswap(compare_out, alternative_val_reg, index_reg)

        else:
            cswap(compare_out, alternative_val_reg, index_reg)
            x(compare_out)
            compare_box(compare_alternative_val_reg, keep_prob_reg, compare_out)
            comparison_prob_prep(compare_alternative_val_reg)

            alternative_val_reg_temp = _unsafe_array_borrow(alternative_val_reg)
            keep_prob_reg_temp = _unsafe_array_borrow(keep_prob_reg)
            alt_and_keep = (alternative_val_reg_temp, keep_prob_reg_temp)

            alias_qrom(index_reg, alt_and_keep)

            _unsafe_array_unborrow(alternative_val_reg, alt_and_keep[0])
            _unsafe_array_unborrow(keep_prob_reg, alt_and_keep[1])
            input_index_prep_dagger(
                index_reg,
            )

    return alias_box


@guppy
@no_type_check
def discard_alias_sampling_garbage[n_index_q: nat, n_keep_q: nat](
    index: array[qubit, n_index_q] @ owned,  # ty: ignore[not-subscriptable]
    alternative: array[qubit, n_index_q] @ owned,  # ty: ignore[not-subscriptable]
    keep: array[qubit, n_keep_q] @ owned,  # ty: ignore[not-subscriptable]
    comparison: array[qubit, n_keep_q] @ owned,  # ty: ignore[not-subscriptable]
    comparison_result: qubit @ owned,
) -> array[qubit, n_index_q]:
    """Discard cleared alias-sampling workspace and retain the index register.

    This helper must be called after alias-sampling UNPREPARE has restored the
    workspace registers to zero. PREPARE followed immediately by UNPREPARE
    satisfies this condition. Inserting SELECT between them generally does not:
    the alias workspace can remain entangled with the system. In that case,
    retain it across walk steps and include it in the preparation-state
    reflection. This function discards qubits; it does not verify zero states.

    Args:
        index: Alias-sampling index register to retain.
        alternative: Cleared alternative-index workspace.
        keep: Cleared alias-sampling acceptance thresholds.
        comparison: Cleared comparison workspace.
        comparison_result: Cleared comparison-result qubit.

    Returns:
        The retained alias-sampling index register.

    """
    discard_array(alternative)
    discard_array(keep)
    discard_array(comparison)
    discard(comparison_result)
    return index


def qrom_bitstrings_from_alias_table(
    alias_table: Aliaser, precision: float
) -> list[tuple[list[bool], list[bool]]]:
    """Calculate qrom input from alias table.

    Due to temporary multi-output qrom, this is subject to change.

    Args:
        alias_table (Aliaser): Alias table to be converted to QROM.
        precision (float): Precision to load keep probabilities.

    """
    n_idx_bits = ceil(log2(len(alias_table)))
    n_prob_bits = ceil(log2(1 / precision))
    alter_color_bits = [
        int_to_bits(int(idx), n_idx_bits) for idx in alias_table.aliases
    ]
    no_alternative_mask = alias_table.get_indices_with_no_alternative()
    keep_prob_bits = [
        float_to_fixed_point(keep_prob, n_prob_bits)
        if idx not in no_alternative_mask
        else [False] * n_prob_bits
        for idx, keep_prob in enumerate(alias_table.probs)
    ]

    return list(zip(alter_color_bits, keep_prob_bits, strict=True))


n_alt = guppy.nat_var("n_alt")
n_keep = guppy.nat_var("n_keep")
AltAndKeep = guppy.type_alias(
    "AltAndKeep",
    "tuple[array[qubit, n_alt], array[qubit, n_keep]]",
    [n_alt, n_keep],
)


def build_alias_fanout[n_alt: nat, n_keep: nat](
    alias_bitstrings: tuple[list[bool], list[bool]],
    fanout_op: GuppyFunctionDefinition = fanout_basic,
) -> GuppyFunctionDefinition[[qubit, AltAndKeep[n_alt, n_keep]], None]:
    """Adapt flat fanout to the differently sized alias and keep registers.

    Alias sampling needs a custom factory because its two target registers may
    have different static lengths. They are joined around one ``fanout_op`` call
    and then split back into the original tuple.

    """
    alt_bits = alias_bitstrings[0]
    keep_bits = alias_bitstrings[1]
    n_alt_qubits = len(alt_bits)
    n_keep_qubits = len(keep_bits)
    n_flat_qubits = n_alt_qubits + n_keep_qubits
    fanout = fanout_from_data(alt_bits + keep_bits, fanout_op)

    @guppy
    @no_type_check
    def alias_fanout(
        control: qubit,
        alt_and_keep: AltAndKeep[n_alt_qubits, n_keep_qubits],
    ) -> None:
        alt_borrowed = _unsafe_array_borrow(alt_and_keep[0])
        keep_borrowed = _unsafe_array_borrow(alt_and_keep[1])
        flattened = join_arrays(
            alt_borrowed,
            keep_borrowed,
            comptime(n_flat_qubits),
        )

        fanout(control, flattened)

        alt_borrowed, keep_borrowed = split_array(
            flattened,
            comptime(n_alt_qubits),
            comptime(n_keep_qubits),
        )
        _unsafe_array_unborrow(alt_and_keep[0], alt_borrowed)
        _unsafe_array_unborrow(alt_and_keep[1], keep_borrowed)

    return alias_fanout
