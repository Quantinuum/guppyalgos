"""Implementation of a unary iteration QROM operation."""

from collections.abc import Callable
from math import ceil, log2
from typing import no_type_check

from guppylang import guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.builtins import array, nat, owned, Function
from guppylang.std.quantum import cz, discard_array, h, measure_array, qubit, x

from guppyalgos.primitives.gate_decompositions.and_op import (
    temp_and,
    uncompute_and,
)
from guppyalgos.primitives.subroutines.fanout import fanout_basic, fanout_from_data
from guppyalgos.algorithms.select import (
    select_unary_iteration,
    build_select_unary_from_data,
)
from guppyalgos.utils import transversal


def qrom_unary_iteration[Data, TargetRegs, n_i_q: nat](
    data_input: list[Data],
    fanout_op: GuppyFunctionDefinition = fanout_basic,
    comp_and_op: GuppyFunctionDefinition[[qubit, qubit, qubit], None] = temp_and,
    uncomp_and_op: GuppyFunctionDefinition[[qubit, qubit, qubit], None] = uncompute_and,
    fanout_from_data_fn: Callable[
        [Data, GuppyFunctionDefinition],
        GuppyFunctionDefinition[[qubit, TargetRegs], None],
    ] = fanout_from_data,
) -> GuppyFunctionDefinition[[array[qubit, n_i_q], TargetRegs], None]:
    """Construct a unary iteration QROM operation.

    Constructs a unary iteration QROM operation based on the provided classical data.

    The construction in figure 5 of https://arxiv.org/pdf/1805.03662 is used as a
    reference. Please see guppyalgos/select/select_unary_iteration.py for more details
    on the unary iteration select operation as this is used as a subroutine to construct
    the QROM operation. For the algorithmic details of the QROM operation please see
    section guppyalgos/select/select_unary_iteration.py.

    There is an option to provide custom fanout, compute AND and uncompute AND
    operations. The default options are basic fanout using CNOTs onto a single target
    register and compute AND using the procedure from https://arxiv.org/pdf/1805.03662
    which uses 4 T gates once the compute ANDs and zero the uncompute AND due to
    measurement based uncomputation.

    The default ``fanout_from_data_fn`` supports one register or equal-sized nested
    registers, flattening nested registers around one ``fanout_op`` call. Supply
    a custom factory for another static shape; for example, alias sampling joins
    and splits two differently sized registers in ``build_alias_fanout``.

    .. code-block:: python3

        import numpy as np
        data_input = [np.random.choice([False, True], size=3).tolist()
            for _ in range(8)]
        qrom = qrom_unary_iteration(data_input)
        @guppy
        @no_type_check
        def main() -> None:
            index_qreg = qarray(2)
            state_qreg = qarray(3)
            qrom(index_qreg, state_qreg)
        main.emulator(n_qubits=5).with_seed(42).with_shots(1).run()


    Args:
        data_input (list[list[bool]]): Classical data to be stored in QROM.
        fanout_op: Flat fanout operation applied to selected data qubits.
        comp_and_op: Function to compute temporary AND operations.
        uncomp_and_op: Function to uncompute temporary AND operations.
        fanout_from_data_fn: Python metafunction adapting ``data_input`` elements and
            ``fanout_op`` to the target type. Override it for heterogeneous
            collections not supported by :func:`fanout_from_data`.

    Returns:
        A Guppy function accepting an index register and the ``TargetRegs``
        shape returned by ``fanout_from_data_fn``.

    """

    def fanout_method(data: Data) -> GuppyFunctionDefinition[[qubit, TargetRegs], None]:
        return fanout_from_data_fn(data, fanout_op)

    return build_select_unary_from_data(
        data_input,
        fanout_method,
        comp_and_op,
        uncomp_and_op,
    )


def qrom_measure_uncompute[n_s_q: nat, n_i_q: nat](
    data_input: list[list[bool]],
    comp_and_op: GuppyFunctionDefinition[[qubit, qubit, qubit], None] = temp_and,
    uncomp_and_op: GuppyFunctionDefinition[[qubit, qubit, qubit], None] = uncompute_and,
) -> GuppyFunctionDefinition[[array[qubit, n_i_q], array[qubit, n_s_q]], None]:
    """QROM measurement based uncompute.

    This works by measuring the state register in the x basis, and using these
    measurement results to classically determine which input states need to have
    a phase applied, which is done via another QROM.
    See figure 5. of https://quantum-journal.org/papers/q-2019-12-02-208

    Currently the cost in terms of gates is very similar to just using the usual QROM
    to uncompute, but the crucial difference is that the output size of the uncomp QROM
    reduced to 1, making it amenable to optimization via selectSWAP.

    TODO missing optimization that can reduce the T count by a factor 2, which involves
    splitting off the last qubit of the index_qreg and tracking which pairs of odd/even
    indices need to be flipped.

    Args:
        data_input (list[list[bool]]): Classical data to be uncomputed
        comp_and_op: Function to compute temporary AND operations.
        uncomp_and_op: Function to uncompute temporary AND operations.

    Returns:
        GuppyFunctionDefinition: Measurement based QROM function, see internal
        measure_uncompute function.

    """
    n_state_qubits = len(data_input[0])
    n_data_elements = len(data_input)
    n_index_qubits = ceil(log2(n_data_elements))

    @guppy.comptime
    @no_type_check
    def data_list_to_array() -> array[array[bool, n_state_qubits], n_data_elements]:
        return array(bools for bools in data_input)

    @guppy
    @no_type_check
    def measure_uncompute(
        index_qreg: array[qubit, n_index_qubits],
        state_qreg: array[qubit, n_state_qubits] @ owned,
    ) -> None:
        """QROM measurement based uncompute.

        Args:
            index_qreg (array[qubit, n_index_qubits]): The
                index qubit register.
            state_qreg (array[qubit, n_state_qubits]): The
                state qubit register to be measured.

        """
        qrom_compute_data = data_list_to_array()

        @guppy
        @no_type_check
        def cz_or_identity(is_cz: bool) -> Function[[qubit, array[qubit, 1]], None]:
            @guppy
            @no_type_check
            def c_identity(control: qubit, target: array[qubit, 1]) -> None:
                pass

            @guppy
            @no_type_check
            def cz_array(control: qubit, target: array[qubit, 1]) -> None:
                cz(control, target[0])

            if is_cz:
                return cz_array
            else:
                return c_identity

        transversal(h, state_qreg)
        ks = measure_array(state_qreg)

        flip_amp_flags = array(False for _ in range(n_data_elements))
        for i in range(len(qrom_compute_data)):
            for j in range(len(qrom_compute_data[i])):
                flip_amp_flags[i] = flip_amp_flags[i] ^ (
                    ks[j].read() & qrom_compute_data[i][j]
                )

        flip_qubit = qubit()
        x(flip_qubit)
        controlled_ops = array(cz_or_identity(bit) for bit in flip_amp_flags)
        fq_array = array(flip_qubit)
        select_unary_iteration(
            controlled_ops, comp_and_op, uncomp_and_op, index_qreg, fq_array
        )
        discard_array(fq_array)

    return measure_uncompute
