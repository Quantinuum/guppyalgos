"""Depth-1 Pauli phasor gadget implementation."""

from __future__ import annotations

from typing import no_type_check

from guppylang import guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.angles import angle
from guppylang.std.builtins import array, comptime, nat, owned
from guppylang.std.option import nothing, some
from guppylang.std.quantum import (
    cx,
    h,
    measure,
    measure_array,
    collect_measurements,
    qubit,
    rz,
    z,
)
from guppyalgos.utils.guppy.gates import transversal, transversal2

import zixy.qubit.pauli as zqp

from guppyalgos.utils import qarray
from guppyalgos.primitives.pauli import pauli_to_z_basis
from guppyalgos.primitives.pauli.pauli import _daggered_basis_change


@guppy
@no_type_check
def _cup[n: nat](ancilla: array[qubit, n]) -> None:
    """Apply the top-half entangling layer used in the depth-1 phasor."""
    if n > 2:
        for idx in range(1, (n + 1) // 2):
            h(ancilla[2 * idx - 1])
            cx(ancilla[2 * idx - 1], ancilla[2 * idx])


@guppy
@no_type_check
def _cap[n: nat](ancilla: array[qubit, n]) -> None:
    """Apply the bottom-half disentangling layer used in the depth-1 phasor."""
    for idx in range(0, n // 2):
        cx(ancilla[2 * idx], ancilla[2 * idx + 1])
        h(ancilla[2 * idx])


@guppy
@no_type_check
def _odd_idx_parity_check[n: nat](data_bits: array[bool, n]) -> bool:
    """Return parity of odd-indexed entries in ``data_bits``."""
    out = False
    for i in range(1, n, 2):
        out ^= data_bits[i]
    return out


@guppy
@no_type_check
def _z_corr_parity[n: nat](data_bits: array[bool, n], start_idx: int) -> bool:
    """Parity helper for feed-forward Z corrections."""
    out = False
    for i in range(2 * start_idx, n, 2):
        out ^= data_bits[i]

    if n % 2 == 0:
        out ^= data_bits[n - 1]

    return out


@guppy
@no_type_check
def _z_correction[n: nat](
    state_q: array[qubit, n],
    ancilla_measures: array[bool, n],
) -> None:
    """Apply Pauli-Z corrections conditioned on ancilla measurement parity."""
    for i in range(n):
        if _z_corr_parity(ancilla_measures, (i + 1) // 2):
            z(state_q[i])


@guppy
@no_type_check
def _z_correction_disconnnected[n: nat, m: nat](
    state_q: array[qubit, n],
    ancilla_measures: array[bool, n],
    ignore_: array[nat, m],
) -> None:
    """Apply Pauli-Z corrections (on non-Identity Paulis).

    Conditioned on ancilla measurement parity.
    """
    for i in range(n):
        apply_gate = True
        for j in range(m):
            if i == ignore_[j]:
                apply_gate = False
        if apply_gate:
            if _z_corr_parity(ancilla_measures, (i + 1) // 2):
                z(state_q[i])


@guppy
@no_type_check
def _append_bool_array[n: nat, m: nat](
    b_arr: array[bool, n] @ owned,  # ty: ignore[not-subscriptable]
    bb: bool,
) -> array[bool, m]:
    """Append one bool to a bool array, preserving ownership semantics."""
    b_arr_opt = array(nothing[bool]() for _ in range(m))

    idx = 0
    for b in b_arr:
        b_arr_opt[idx].swap(some(b)).unwrap_nothing()
        idx += 1

    b_arr_opt[m - 1].swap(some(bb)).unwrap_nothing()
    return array(b.unwrap() for b in b_arr_opt)


def pauli_exp_depth1[n_state_q: nat](
    pauli_string: zqp.String,
    n_qubits: int,
    rz_method: GuppyFunctionDefinition[[qubit, angle], None] = rz,
) -> GuppyFunctionDefinition[[array[qubit, n_state_q], angle], None]:
    r"""Generate a depth-1 Pauli phasor gadget for a Pauli string.

    This implements the measurement-and-correction construction from Fig. 7 of
    arXiv:2603.17774, using a basis-change into Z, a depth-1 parity extraction on
    ancillae, a parity-conditioned phase, and feed-forward Z corrections.

    Args:
        pauli_string (rqp.String): zixy string to exponentiate.
        n_qubits: total register size.
        rz_method (GuppyFunctionDefinition): single-qubit phase method used for
            the terminal phase rotation.

    Returns:
        Guppy function implementing ``exp(-i * theta/2 * P)`` on the input register.

    """
    if pauli_string.is_identity():
        raise ValueError(
            "Depth-1 Pauli phasor requires at least 1 non-identity Pauli operator"
        )

    basis_change = pauli_to_z_basis(pauli_string=pauli_string, size=n_qubits)
    restore_basis = _daggered_basis_change(basis_change, n_qubits)

    id_indices = [i for i in range(n_qubits) if pauli_string[i] == zqp.I]

    if len(id_indices) > 0:

        @guppy
        @no_type_check
        def pauli_phasor_fn(
            qreg: array[qubit, comptime(n_qubits)],
            theta: angle,
        ) -> None:
            ancilla_qreg = qarray(comptime(n_qubits))
            basis_change(qreg)
            _cup(ancilla_qreg)

            transversal2(
                cx, qreg, ancilla_qreg, array(nat(el) for el in comptime(id_indices))
            )
            _cap(ancilla_qreg)
            *qa, qb = ancilla_qreg
            res_a = collect_measurements(measure_array(qa))
            if _odd_idx_parity_check(res_a):
                rz_method(qb, -theta)
            else:
                rz_method(qb, theta)
            h(qb)
            res_b = measure(qb).read()
            result_ancilla: array[bool, comptime(n_qubits)] = _append_bool_array(
                res_a, res_b
            )
            _z_correction_disconnnected(
                qreg, result_ancilla, array(nat(el) for el in comptime(id_indices))
            )
            restore_basis(qreg)

        return pauli_phasor_fn
    else:

        @guppy
        @no_type_check
        def pauli_phasor_fn(
            qreg: array[qubit, comptime(n_qubits)],
            theta: angle,
        ) -> None:
            ancilla_qreg = qarray(comptime(n_qubits))
            basis_change(qreg)
            _cup(ancilla_qreg)
            transversal(cx, qreg, ancilla_qreg)
            _cap(ancilla_qreg)
            *qa, qb = ancilla_qreg
            res_a = collect_measurements(measure_array(qa))
            if _odd_idx_parity_check(res_a):
                rz_method(qb, -theta)
            else:
                rz_method(qb, theta)
            h(qb)
            res_b = measure(qb).read()
            result_ancilla: array[bool, comptime(n_qubits)] = _append_bool_array(
                res_a, res_b
            )
            _z_correction(qreg, result_ancilla)
            restore_basis(qreg)

        return pauli_phasor_fn
