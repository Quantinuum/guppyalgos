"""Ladders of multi-controlled X gates."""

from typing import no_type_check

from guppylang import guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.builtins import array
from guppylang.std.quantum import qubit

from guppyalgos.primitives.gate_decompositions.cnx.cnx_cca import cnx_cca_logdepth_dirty
from guppyalgos.primitives.subroutines.ladders.toffoli_ladder import (
    ToffoliLadderLog,
    log_ladder_ascending_dagger,
    log_ladder_ascending_dagger_with_cca,
    log_toffoli_ladder_num_ancilla,
)


def cnx_ladder_logdepth_num_ancilla(k: int) -> int:
    """Count the clean ancillae the log depth cnx ladder needs.

    Args:
        k: Length of the ladder.

    """
    if k < 1:
        raise ValueError("The ladder needs at least one gate")
    if k <= 3:
        return 0
    n_middle = (k + 1) // 2 - 1
    return n_middle + log_toffoli_ladder_num_ancilla(2 * n_middle + 1)


def cnx_ladder_logdepth(
    k: int, n: int, inverse: bool = False
) -> GuppyFunctionDefinition:
    """Build a ladder of k C^nX gates in logarithmic depth.

    Args:
        k: Length of the ladder.
        n: Number of controls of each gate.
        inverse: Apply the reverse gate sequence.

    Returns:
        A Guppy function implementing the C^nX ladder in logarithmic depth.

    """
    if k < 1:
        raise ValueError("The ladder needs at least one gate")
    n_controls_b = n - 1
    n_ancillae = cnx_ladder_logdepth_num_ancilla(k)

    @guppy.comptime
    @no_type_check
    def cnx_ladder_logdepth_impl(
        controls_a: array[qubit, k],
        controls_b: array[array[qubit, n_controls_b], k],
        target: qubit,
        borrowed_a: array[qubit, k],
        borrowed_b: array[qubit, k],
        ancillae: array[qubit, n_ancillae],
    ) -> None:
        """Apply the ladder of multi-controlled X gates.

        Args:
            controls_a (array[qubit, k]): First control of each gate; also the
                target of the previous one.
            controls_b (array[array[qubit, n - 1], k]): Register of the n - 1
                remaining controls of each gate.
            target (qubit): Target of the last gate.
            borrowed_a (array[qubit, k]): First borrowed workspace qubit of
                each gate.
            borrowed_b (array[qubit, k]): Second borrowed workspace qubit of
                each gate.
            ancillae (array[qubit, cnx_ladder_logdepth_num_ancilla(k)]): Clean
                ancillae to use.

        """
        chain = list(controls_a)
        targets = [*chain[1:], target]
        gates = [
            (chain[i], list(controls_b[i]), targets[i], borrowed_a[i], borrowed_b[i])
            for i in range(k)
        ]

        def apply_cnx_gates(group_of_gates) -> None:
            """Apply one group of multi-controlled X gates."""
            for chain_wire, group, tgt, dirty_a, dirty_b in group_of_gates:
                cnx_cca_logdepth_dirty([chain_wire, *group], tgt, dirty_a, dirty_b)

        if k <= 3:
            apply_cnx_gates(reversed(gates) if inverse else gates)
            return

        even, odd = gates[0::2], gates[1::2]
        taken = iter(list(ancillae))
        and_ancillae = [next(taken) for _ in range(len(even) - 1)]

        def apply_and() -> None:
            """AND the groups of each adjacent pair onto its ancilla."""
            for j, ancilla in enumerate(and_ancillae):
                _, group, _, dirty_a, dirty_b = odd[j]
                cnx_cca_logdepth_dirty(
                    [*group, *even[j + 1][1]], ancilla, dirty_a, dirty_b
                )

        first, second = (odd, even) if inverse else (even, odd)
        apply_cnx_gates(first)
        apply_and()
        mid_a = [gate[2] for gate in even[:-1]]
        middle = [q for pair in zip(mid_a, and_ancillae, strict=True) for q in pair]
        middle.append(even[-1][2])
        ladder = ToffoliLadderLog()
        rest = list(taken)
        if inverse:
            if rest:
                log_ladder_ascending_dagger_with_cca(middle, rest)
            else:
                log_ladder_ascending_dagger(middle)
        elif rest:
            ladder.ascending_with_cca(middle, rest)
        else:
            ladder.ascending(middle)
        apply_and()
        apply_cnx_gates(second)

    return cnx_ladder_logdepth_impl
