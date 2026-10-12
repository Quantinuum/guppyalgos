"""Uniform state preparation for quantum algorithms."""

from guppylang import guppy
from guppylang.std.builtins import array, nat
from guppylang.std.quantum import qubit, h, x, rz, discard
from guppylang.std.angles import angle
from guppylang.defs import GuppyFunctionDefinition
from guppyalgos.primitives.gate_decompositions.cnx.cnx import cnx
from guppyalgos.primitives.measurement.utils import discard_array
from guppyalgos.utils import int_to_bits, transversal, apply_bitstring, qarray
import numpy as np
from guppyalgos.primitives.arithmetic.comparator import (
    comparator_ripple_cuccaro,
)

from typing import no_type_check


def uniform_state[n: nat](
    num_nonzero_amplitudes: int,
    cnx_box: GuppyFunctionDefinition[[array[qubit, n], qubit], None] = cnx,
    comparator_box: GuppyFunctionDefinition[
        [array[qubit, n], array[qubit, n], qubit], None
    ] = comparator_ripple_cuccaro,
    dagger: bool = False,
) -> GuppyFunctionDefinition[[array[qubit, n]], None]:
    r"""Prepare a uniform state over the first `num_nonzero_amplitudes` basis states.

    This unitary will act on ``ceil(log2(num_nonzero_amplitudes))`` qubits.

    The output state is given by:
    $$\frac{1}{\sqrt{L}} \sum_{i=0}^{L-1} \ket{i}$$
    where $L$= ``num_nonzero_amplitudes``

    This function prepares a uniform superposition state on the input qubits.
    The input qubits are assumed to be initialized to $\ket{0}$ state.
    The function first checks if the number of non-zero amplitudes
    is equal to $2^n$. If equal, a Hadamard transformation is applied.
    Otherwise, an amplitude amplification based circuit is used.

    Args:
        num_nonzero_amplitudes (int):
                The number of non-zero amplitudes of the uniform state.
        cnx_box (GuppyFunctionDefinition, optional):
                The multi-controlled CNOT function. Defaults to `cnx`.
        comparator_box (GuppyFunctionDefinition, optional):
                The comparator function. Defaults to `ripple_carry_subtractor`.
        dagger (bool):
            Return the dagger of the state prep if True, only relevant for the case
            where `num_nonzero_amplitudes` is not a power of 2.

    Returns:
        GuppyFunctionDefinition: A Guppy function that prepares the uniform state
        on `ceil(log2(L))` qubits.

    Notes:
        - If `num_nonzero_amplitudes` is equal to $2^n$, the function applies a
          Hadamard transformation to prepare the uniform state.
        - If `num_nonzero_amplitudes` is not a power of 2,
            additional ancilla qubits will be required for applying amplitude
            amplification. (minimum 1, but currently n+1)

    """
    n_qubit = int(np.ceil(np.log2(num_nonzero_amplitudes)))
    uniform_function_checks = np.log2(num_nonzero_amplitudes) == n_qubit
    if uniform_function_checks:

        @guppy
        @no_type_check
        def uniform_state_div(q: array[qubit, n_qubit]) -> None:
            r"""Prepare when state with non zero amplitude number is 2^n.

            This function prepares a uniform superposition state on the input qubits.
            The input qubits are assumed to be initialized to |0⟩ state.

            Args:
                q (array[qubit, n]): Array of qubits to prepare the uniform state on.

            """
            transversal(h, q)

        return uniform_state_div

    else:
        L_bit_array = int_to_bits(num_nonzero_amplitudes, n_qubit)
        _rz_angle = (
            np.arccos(
                (num_nonzero_amplitudes - 2**n_qubit / 2) / num_nonzero_amplitudes
            )
            / np.pi
        )

        @guppy
        @no_type_check
        def uniform_state_non_div_box(
            q_reg: array[qubit, n_qubit],
        ) -> None:
            r"""Prepare when non zero amplitude number is not divisible by 2^n.

            This function prepares a uniform superposition state on the input qubits.
            The input qubits are assumed to be initialized to |0⟩ state.
            * Note default function requires additional logL+3+\ceil{(logL-3)/2} qubits.

            Args:
                q_reg: The quantum register with log(L) qubits.

            """
            rz_angle = angle(_rz_angle)
            anci_reg_L = qarray(n_qubit)
            ancilla = qubit()

            if not dagger:
                transversal(h, q_reg)

                apply_bitstring(anci_reg_L, L_bit_array)

                comparator_box(q_reg, anci_reg_L, ancilla)
                rz(ancilla, rz_angle)
                comparator_box(q_reg, anci_reg_L, ancilla)

                transversal(h, q_reg)

                transversal(x, q_reg)
                cnx_box(q_reg, ancilla)
                rz(ancilla, rz_angle)
                cnx_box(q_reg, ancilla)
                transversal(x, q_reg)

                transversal(h, q_reg)
                apply_bitstring(anci_reg_L, L_bit_array)

            else:
                apply_bitstring(anci_reg_L, L_bit_array)
                transversal(h, q_reg)

                transversal(x, q_reg)
                cnx_box(q_reg, ancilla)
                rz(ancilla, -rz_angle)
                cnx_box(q_reg, ancilla)
                transversal(x, q_reg)

                transversal(h, q_reg)

                comparator_box(q_reg, anci_reg_L, ancilla)
                rz(ancilla, -rz_angle)
                comparator_box(q_reg, anci_reg_L, ancilla)

                apply_bitstring(anci_reg_L, L_bit_array)
                transversal(h, q_reg)

            discard_array(anci_reg_L)
            discard(ancilla)

        return uniform_state_non_div_box
