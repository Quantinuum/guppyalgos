"""Implementation of a unary iteration QROM operation."""

from collections.abc import Callable

from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.builtins import array, nat
from guppylang.std.quantum import qubit

from guppyalgos.primitives.gate_decompositions.and_op import (
    temp_and,
)
from guppyalgos.primitives.subroutines.fanout import fanout_basic, fanout_from_data
from guppyalgos.algorithms.select import (
    build_cntrl_select_unary_from_data,
)


def cntrl_qrom_unary_iteration[Data, TargetRegs, n_i_q: nat](
    data_input: list[Data],
    fanout_op: GuppyFunctionDefinition = fanout_basic,
    and_op: GuppyFunctionDefinition[[qubit, qubit, qubit], None] = temp_and,
    fanout_from_data_fn: Callable[
        [Data, GuppyFunctionDefinition],
        GuppyFunctionDefinition[[qubit, TargetRegs], None],
    ] = fanout_from_data,
) -> GuppyFunctionDefinition[[qubit, array[qubit, n_i_q], TargetRegs], None]:
    """Construct a controlled unary iteration QROM operation.

    This meta function is an extension of the QROM unary iteration function where
    the unary iteration is controlled by an additional control qubit.
    When the control qubit is in the ``|1>`` state, the QROM operation is applied as
    normal. When the control qubit is in the ``|0>`` state, the QROM operation is not
    applied
    and the state register remains unchanged.

    Constructs a controlled unary iteration QROM operation based on the provided
    classical data.

    The construction in figure 5 of https://arxiv.org/pdf/1805.03662 is used as a
    reference. Please see guppyalgos/select/select_unary_iteration.py for more details
    on the unary iteration select operation as this is used as a subroutine to construct
    the QROM operation. For the algorithmic details of the QROM operation please see
    section guppyalgos/select/select_unary_iteration.py.

    There is an option to provide a custom fanout and an AND unitary.
    Uncomputation uses the AND unitary's dagger. The defaults are CNOT fanout
    and the temporary AND from https://arxiv.org/pdf/1805.03662, which uses four
    T gates for computation and none for measurement-based uncomputation.

    The default factory maps a bitstring to one register. Supply a custom
    ``fanout_from_data_fn`` when the data uses another static target shape, such as
    alias sampling's pair of differently sized registers.

    .. code-block:: python3

        import numpy as np
        data_input = [np.random.choice([False, True], size=3).tolist()
            for _ in range(8)]
        qrom = cntrl_qrom_unary_iteration(data_input)
        @guppy
        @no_type_check
        def main() -> None:
            index_qreg = qarray(2)
            state_qreg = qarray(3)
            qrom(index_qreg, state_qreg)
        main.emulator(n_qubits=5).with_seed(42).with_shots(1).run()


    Args:
        data_input: Classical data used to construct the controlled operations.
        fanout_op: Flat Guppy fanout operation applied to selected data qubits.
        and_op: AND unitary whose dagger uncomputes each work qubit.
        fanout_from_data_fn: Python metafunction adapting a data element and
            ``fanout_op`` to the target type.

    Returns:
        A Guppy function accepting a control qubit, index register, and the
        ``TargetRegs`` shape returned by ``fanout_from_data_fn``.

    """

    def fanout_method(data: Data) -> GuppyFunctionDefinition[[qubit, TargetRegs], None]:
        return fanout_from_data_fn(data, fanout_op)

    return build_cntrl_select_unary_from_data(
        data_input,
        fanout_method,
        and_op,
    )
