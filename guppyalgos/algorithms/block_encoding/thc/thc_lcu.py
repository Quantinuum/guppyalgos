"""Controlled LCU and qubitization walk helpers for THC algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log2
from typing import no_type_check

import numpy as np
import numpy.typing as npt
from guppylang import guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.builtins import Function, array, comptime, nat
from guppylang.std.quantum import qubit

from guppyalgos.algorithms.select.qrom import qrom_unary_iteration
from guppyalgos.utils.guppy.array import join_arrays, split_array

from .thc_helpers import (
    THCParameters,
    build_thc_alias_terms,
    build_select_data,
    encode_combined_givens_rotations,
    encode_givens_rotations,
    validate_thc_parameters,
)
from .thc_select import SelectTHCCntrlRegs


@dataclass(frozen=True)
class THCData:
    """Classical data and Guppy QROMs used to construct a controlled THC LCU.

    This value contains only the result of classical preprocessing: normalized
    coefficient magnitudes, encoded QROM tables, and the dimensions needed to
    allocate their registers. It does not choose a quantum rotation synthesis
    method or construct PREPARE, SELECT, and UNPREPARE.

    In particular, the QROMs encode rotation angles without depending on the
    rotator that consumes them. A caller may combine them with a phase-gradient
    Givens cascade, a resource-free cascade, or any other signature-compatible
    Select implementation.
    """

    alias_probabilities: npt.NDArray[np.float64]
    alias_precision: float
    select_data_loader: GuppyFunctionDefinition
    qrom_1_and_2_body: GuppyFunctionDefinition
    qrom_2_body: GuppyFunctionDefinition
    n_alias_qubits: int
    n_index_qubits: int
    n_keep_qubits: int
    n_modes: int
    n_givens: int
    rotation_precision_bits: int


@guppy
@no_type_check
def load_select_registers[n_alias_q: nat, n_index_q: nat, n_select_data: nat](
    select_data_loader: Function[
        [
            array[qubit, n_alias_q],
            array[qubit, n_select_data],
        ],
        None,
    ],
    alias_index: array[qubit, n_alias_q],
    select_regs: SelectTHCCntrlRegs[n_index_q],
) -> None:
    """Toggle Select fields using the supplied data loader and alias index.

    The QROM stores records of the form ``[mu, nu, one_body, sign]``. This
    function temporarily joins the corresponding Select fields, applies the
    QROM, and splits them back into their named registers. Calling it again with
    the same alias index clears the fields during UNPREPARE.

    Args:
        select_data_loader: QROM loading one flattened Select record.
        alias_index: Address of the sampled THC term.
        select_regs: Named THC Select fields loaded by the QROM.

    """
    flags = join_arrays(
        array(select_regs.one_body_flag),
        array(select_regs.coefficient_sign),
        2,
    )
    select_tail = join_arrays(
        select_regs.second_index_qreg,
        flags,
        comptime(n_index_q + 2),
    )
    select_data = join_arrays(
        select_regs.first_index_qreg,
        select_tail,
        n_select_data,
    )
    select_data_loader(alias_index, select_data)
    select_regs.first_index_qreg, select_tail = split_array(
        select_data,
        n_index_q,
        comptime(n_index_q + 2),
    )
    select_regs.second_index_qreg, flags = split_array(
        select_tail,
        n_index_q,
        2,
    )
    one_body_flag_qreg, coefficient_sign_qreg = split_array(flags, 1, 1)
    select_regs.one_body_flag = one_body_flag_qreg.take(0)
    select_regs.coefficient_sign = coefficient_sign_qreg.take(0)
    one_body_flag_qreg.discard_all_taken()
    coefficient_sign_qreg.discard_all_taken()


def build_thc_lcu_data(
    parameters: THCParameters,
    *,
    rotation_precision_bits: int,
    alias_precision_bits: int,
) -> THCData:
    """Classically preprocess THC parameters for a controlled LCU.

    This function validates the input data, normalizes the THC coefficient
    magnitudes, encodes the Select records and Givens angles, and builds Guppy
    QROM callables for those tables. It deliberately stops at that classical
    data boundary: it does not allocate qubits, prepare resource states, or
    instantiate PREPARE, SELECT, UNPREPARE, or a rotation cascade.

    Consequently, the returned data are agnostic to the quantum Select and
    rotation implementation. For example, the QROM angle tables can feed a
    phase-gradient Givens cascade, but phase-gradient preparation and ownership
    belong to the caller. Another compatible rotator can consume the same data
    without changing this preprocessing function.

    The generated Givens-angle QROM entries use the parallel data-register
    layout: each entry contains ``n_givens`` distinct little-endian binary angle
    registers, each of width ``rotation_precision_bits``. A cascade therefore
    receives one data register for each neighboring Givens rotation, rather
    than reusing a single register to load its angles serially. This uses more
    angle-register qubits in exchange for the corresponding depth-efficient
    data-loading interface.

    Args:
        parameters: Classical THC coefficients and neighboring-Givens angles.
        rotation_precision_bits: Number of bits used to encode each rotation.
        alias_precision_bits: Number of bits used by alias-sampling probabilities.

    Returns:
        Normalized alias data, encoded Guppy QROMs, and dimensions from which a
        caller can construct its chosen controlled LCU.

    """
    n_modes, thc_rank = validate_thc_parameters(parameters)
    if rotation_precision_bits <= 0:
        raise ValueError("rotation_precision_bits must be positive.")
    if alias_precision_bits <= 0:
        raise ValueError("alias_precision_bits must be positive.")

    n_givens = n_modes - 1
    n_index_qubits = ceil(log2(max(n_modes, thc_rank + 1)))
    terms = build_thc_alias_terms(parameters)
    n_alias_qubits = ceil(log2(len(terms)))
    alias_precision = 2.0**-alias_precision_bits
    weights = np.asarray([abs(term.coefficient) for term in terms], dtype=np.float64)
    if not np.any(weights):
        raise ValueError("At least one THC coefficient must be non-zero.")

    alias_probabilities = weights / np.sum(weights)
    select_data_loader = qrom_unary_iteration(
        build_select_data(terms, n_index_qubits, thc_rank)
    )
    # The first QROM contains both the two-body U_mu and one-body V_mu rotations.
    qrom_1_and_2_body = qrom_unary_iteration(
        encode_combined_givens_rotations(
            parameters.two_body_rotations,
            parameters.one_body_rotations,
            rotation_precision_bits,
            n_index_qubits,
        )
    )
    qrom_2_body_data = encode_givens_rotations(
        parameters.two_body_rotations,
        rotation_precision_bits,
        2**n_index_qubits,
    )
    # Give the QROM the same address width as the loaded nu register. PREPARE never
    # selects the appended rows, so their copied values do not affect the encoded LCU.
    minimum_qrom_rows = (2 ** (n_index_qubits - 1)) + 1
    qrom_2_body_data.extend(
        qrom_2_body_data[0] for _ in range(minimum_qrom_rows - len(qrom_2_body_data))
    )
    qrom_2_body = qrom_unary_iteration(qrom_2_body_data)
    return THCData(
        alias_probabilities,
        alias_precision,
        select_data_loader,
        qrom_1_and_2_body,
        qrom_2_body,
        n_alias_qubits,
        n_index_qubits,
        alias_precision_bits,
        n_modes,
        n_givens,
        rotation_precision_bits,
    )
