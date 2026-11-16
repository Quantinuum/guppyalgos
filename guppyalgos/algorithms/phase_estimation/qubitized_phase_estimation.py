"""Canonical phase-estimation registers for qubitization."""

from typing import no_type_check

from guppylang import guppy
from guppylang.std.builtins import Unitary, array, control, dagger, nat
from guppylang.std.quantum import qubit


@guppy.struct
class QubitizationRegs[n_prep_q: nat, TargetRegs]:
    """Registers acted on by a qubitized walk operator during phase estimation."""

    prep: array[qubit, n_prep_q]
    target: TargetRegs


@guppy.unitary
class qubitized_power_oracle:  # noqa: D101
    @guppy
    @no_type_check
    def __call__[n_prep_q: nat, n_target_q: nat](
        control_qubit: qubit,
        prep: array[qubit, n_prep_q],
        target: array[qubit, n_target_q],
        power: int,
        cntrl_walk: Unitary[
            [qubit, array[qubit, n_prep_q], array[qubit, n_target_q]], None
        ],
    ) -> None:
        """Apply a controlled qubitization walk ``power`` times.

        ``cntrl_walk`` is a ``Unitary`` callback taking an intrinsic control,
        a preparation array, and a target array. Register sizes are inferred;
        all quantum arguments are borrowed and restored in place. Nonpositive
        runtime powers apply no steps.

        Public ``control`` and ``dagger`` contexts use explicit implementations.
        Inverse paths traverse repetitions in descending order and invert each
        walk step. Repetitions of the same walk have identical step bodies.

        Pass separate array variables inside modifier contexts: Guppy 1.2.0
        cannot normalize repacking borrowed struct fields in those contexts.
        For example::

            with control(external), dagger:
                qubitized_power_oracle(c, prep, target, power, cntrl_walk)

        Public struct modifier migration remains separate.
        """
        for _ in range(power):
            cntrl_walk(control_qubit, prep, target)

    @guppy
    @no_type_check
    def daggered[n_prep_q: nat, n_target_q: nat](
        control_qubit: qubit,
        prep: array[qubit, n_prep_q],
        target: array[qubit, n_target_q],
        power: int,
        cntrl_walk: Unitary[
            [qubit, array[qubit, n_prep_q], array[qubit, n_target_q]], None
        ],
    ) -> None:
        """Apply inverse walk steps in reverse repetition order."""
        for _ in range(power, 0, -1):
            with dagger:
                cntrl_walk(control_qubit, prep, target)

    @guppy
    @no_type_check
    def controlled[n_prep_q: nat, n_target_q: nat, n_ctrl: nat](
        control_qubit: qubit,
        prep: array[qubit, n_prep_q],
        target: array[qubit, n_target_q],
        power: int,
        cntrl_walk: Unitary[
            [qubit, array[qubit, n_prep_q], array[qubit, n_target_q]], None
        ],
        ctrls: array[qubit, n_ctrl],
    ) -> None:
        """Control every repetition with the external controls."""
        for _ in range(power):
            with control(ctrls):
                cntrl_walk(control_qubit, prep, target)

    @guppy
    @no_type_check
    def ctrl_daggered[n_prep_q: nat, n_target_q: nat, n_ctrl: nat](
        control_qubit: qubit,
        prep: array[qubit, n_prep_q],
        target: array[qubit, n_target_q],
        power: int,
        cntrl_walk: Unitary[
            [qubit, array[qubit, n_prep_q], array[qubit, n_target_q]], None
        ],
        ctrls: array[qubit, n_ctrl],
    ) -> None:
        """Apply controlled inverse steps in reverse repetition order."""
        for _ in range(power, 0, -1):
            with control(ctrls):
                with dagger:
                    cntrl_walk(control_qubit, prep, target)
