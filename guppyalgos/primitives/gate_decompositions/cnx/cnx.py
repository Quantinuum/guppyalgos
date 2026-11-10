"""Multicontrolled x gate."""

from __future__ import annotations

from typing import no_type_check

from guppylang.decorator import guppy
from guppylang.std.builtins import array, comptime, nat, owned
from guppylang.std.quantum import (
    cx,
    discard,
    discard_array,
    h,
    qubit,
    t,
    tdg,
    toffoli,
    x,
)


from guppyalgos.primitives.gate_decompositions.and_op import (
    temp_and_compute,
    temp_and_uncompute,
)
from guppyalgos.utils import qarray


@guppy
@no_type_check
def cnx[n_controls: nat](control: array[qubit, n_controls], target: qubit) -> None:
    r"""Apply efficient cnx in terms of number of 2q gates.

    Based on https://arxiv.org/pdf/1508.03273, given $n - 1$ control qubits and the
    target, uses $\lceil{(n-3)/2}\rceil$ ancillary qubits to apply the multicontrolled-x
    gate on the target using $6n -12$ CNOT gates.

    Args:
        control (array[qubit, n_controls]): register of control qubits.
        target (qubit): target qubit.

    """
    is_odd, num_ancillas = _get_variables_cnx(n_controls)
    if n_controls == 0:
        x(target)
    elif n_controls == 1:
        cx(control[0], target)
    elif n_controls == 2:
        toffoli(control[0], control[1], target)
    elif n_controls == 3:
        ancilla = qubit()
        _tof4(control[0], control[1], control[2], target, ancilla)
        discard(ancilla)
    elif n_controls == 4:
        ancilla = qubit()
        _tof5(control[0], control[1], control[2], control[3], target, ancilla)
        discard(ancilla)
    else:
        ancilla = qubit()
        _rtof4(control[0], control[1], control[2], ancilla)
        _cnx_aux(control, target, ancilla, num_ancillas, 3, 0, is_odd)
        _irtof4(control[0], control[1], control[2], ancilla)
        discard(ancilla)


@guppy
@no_type_check
def cnx_single_ancilla[n_controls: nat](
    control: array[qubit, n_controls], target: qubit
) -> None:
    """Cnx requiring only a single ancilla.

    Works by splitting the cnx into two cnx's of half size using one ancilla and
    borrowing qubits from one half in perform the cnx for the other half.
    Uses ~12*n_control toffoli gates.

    Args:
        control (array[qubit, n_controls]): register of control qubits.
        target (qubit): target qubit.

    """
    if n_controls == 0:
        x(target)
    elif n_controls == 1:
        cx(control[0], target)
    elif n_controls == 2:
        toffoli(control[0], control[1], target)
    elif n_controls == 3:
        ancilla = qubit()
        _tof4(control[0], control[1], control[2], target, ancilla)
        discard(ancilla)
    elif n_controls == 4:
        ancilla = qubit()
        _tof5(control[0], control[1], control[2], control[3], target, ancilla)
        discard(ancilla)
    else:
        _cnx_single_ancilla_aux(control, target)


@guppy
@no_type_check
def cnx_toffoli_ladder[n_controls: nat](
    control: array[qubit, n_controls], target: qubit
) -> None:
    """Cnx using n_controls - 2 zeroed ancillas.

    Applies the standard temporary-AND ladder construction with
    measurement-based uncompute for the temporary-ANDs (relative toffolis).

    Args:
        control (array[qubit, n_controls]): register of control qubits.
        target (qubit): target qubit.

    """
    work = CNXPartialAndLadder.compute(control, target)
    CNXPartialAndLadder.uncompute_workspace(control, work)


@guppy.struct
class CNXPartialAndLadder:  # noqa: D101
    @guppy
    @no_type_check
    @staticmethod
    def compute[n: nat](
        controls: array[qubit, n], target: qubit
    ) -> array[qubit, comptime(max(n - 2, 0))]:  # ty:ignore[unsupported-operator]
        """Compute AND of controls onto target using temp AND ladder.

        returns ancilla workspace to be uncomputed by `uncompute_workspace` or
        `uncompute`.
        """
        work = qarray(comptime(max(n - 2, 0)))
        if n > 2:
            temp_and_compute(controls[0], controls[1], work[0])
            for index in range(n - 3):
                temp_and_compute(work[index], controls[index + 2], work[index + 1])

        CNXPartialAndLadder._final_gate(controls, work, target)
        return work

    @guppy
    @no_type_check
    @staticmethod
    def uncompute_workspace[n: nat](
        controls: array[qubit, n],
        work: array[qubit, comptime(max(n - 2, 0))] @ owned,  # ty: ignore[not-subscriptable, unsupported-operator]
    ) -> None:
        """Uncompute ancilla workspace, keeping result of AND in target."""
        if n > 2:
            for index in range(n - 4, -1, -1):
                temp_and_uncompute(work[index], controls[index + 2], work[index + 1])
            temp_and_uncompute(controls[0], controls[1], work[0])
        discard_array(work)

    @guppy
    @no_type_check
    @staticmethod
    def uncompute[n: nat](
        controls: array[qubit, n],
        work: array[qubit, comptime(max(n - 2, 0))] @ owned,  # ty: ignore[not-subscriptable, unsupported-operator]
        target: qubit,
    ) -> None:
        """Uncompute target and ancilla workspace."""
        CNXPartialAndLadder._final_gate(controls, work, target)
        CNXPartialAndLadder.uncompute_workspace(controls, work)

    @guppy
    @no_type_check
    @staticmethod
    def _final_gate[n: nat](
        controls: array[qubit, n],
        work: array[qubit, comptime(max(n - 2, 0))],  # ty: ignore[unsupported-operator]
        target: qubit,
    ) -> None:
        if n == 0:
            x(target)
        elif n == 1:
            cx(controls[0], target)
        elif n == 2:
            toffoli(controls[0], controls[1], target)
        else:
            toffoli(work[n - 3], controls[n - 1], target)


@guppy
@no_type_check
def _rtof4(a: qubit, b: qubit, c: qubit, d: qubit) -> None:
    """Apply the relative tof4 block."""
    h(d)
    t(d)
    cx(c, d)
    tdg(d)
    h(d)
    cx(a, d)
    t(d)
    cx(b, d)
    tdg(d)
    cx(a, d)
    t(d)
    cx(b, d)
    tdg(d)
    h(d)
    t(d)
    cx(c, d)
    tdg(d)
    h(d)


@guppy
@no_type_check
def _irtof4(a: qubit, b: qubit, c: qubit, d: qubit) -> None:
    """Apply the inverse of relative tof4 block."""
    h(d)
    t(d)
    cx(c, d)
    tdg(d)
    h(d)
    t(d)
    cx(b, d)
    tdg(d)
    cx(a, d)
    t(d)
    cx(b, d)
    tdg(d)
    cx(a, d)
    h(d)
    t(d)
    cx(c, d)
    tdg(d)
    h(d)


@guppy
@no_type_check
def _tof4(
    control0: qubit,
    control1: qubit,
    control2: qubit,
    target: qubit,
    ancilla: qubit,
) -> None:
    """Apply tof4 block."""
    temp_and_compute(control0, control1, ancilla)
    toffoli(ancilla, control2, target)
    temp_and_uncompute(control0, control1, ancilla)


@guppy
@no_type_check
def _tof5(
    control0: qubit,
    control1: qubit,
    control2: qubit,
    control3: qubit,
    target: qubit,
    ancilla: qubit,
) -> None:
    """Apply tof5 block."""
    _rtof4(control0, control1, control2, ancilla)
    toffoli(ancilla, control3, target)
    _irtof4(control0, control1, control2, ancilla)


@guppy.comptime
@no_type_check
def _get_variables_cnx(n_controls: int) -> tuple[int, int]:
    """Auxiliary method for cnx that computes the necessary variables."""
    is_odd = n_controls % 2
    num_ancillas = (n_controls - 2) // 2 + (n_controls - 2) % 2
    return is_odd, num_ancillas


@guppy
@no_type_check
def _cnx_aux[n_controls: nat](
    control: array[qubit, n_controls],
    target: qubit,
    ancilla: qubit,
    n_ancillas: int,
    control_counter: int,
    ancilla_counter: int,
    is_odd: int,
) -> None:
    """Auxiliary method for cnx that recursively builds circuit."""
    new_ancilla = qubit()
    if ancilla_counter == n_ancillas - 2:
        if is_odd == 1:
            _tof4(
                ancilla,
                control[n_controls - 2],
                control[n_controls - 1],
                target,
                new_ancilla,
            )
        else:
            _tof5(
                ancilla,
                control[n_controls - 3],
                control[n_controls - 2],
                control[n_controls - 1],
                target,
                new_ancilla,
            )

    else:
        _rtof4(
            ancilla, control[control_counter], control[control_counter + 1], new_ancilla
        )
        _cnx_aux(
            control,
            target,
            new_ancilla,
            n_ancillas,
            control_counter + 2,
            ancilla_counter + 1,
            is_odd,
        )
        _irtof4(
            ancilla, control[control_counter], control[control_counter + 1], new_ancilla
        )
    discard(new_ancilla)


@guppy
@no_type_check
def _cnx_single_ancilla_aux[n_controls: nat](
    control: array[qubit, n_controls], target: qubit
) -> None:
    """Single ancilla cnx.

    Described in
    https://algassert.com/circuits/2015/06/05/Constructing-Large-Controlled-Nots.html
    """
    ancilla = qubit()
    _cnx_borrowed_ancilla(control, ancilla)
    _cnx_borrowed_target(control, target, ancilla)
    _cnx_borrowed_ancilla(control, ancilla)
    discard(ancilla)


@guppy
@no_type_check
def _cnx_borrowed_ancilla[n_controls: nat](
    control: array[qubit, n_controls], ancilla: qubit
) -> None:
    n_controls = len(control)
    odd_controls = (n_controls % 2) == 1
    top_idx = n_controls - 1 if odd_controls else n_controls - 2

    for _ in range(2):
        toffoli(control[top_idx - 1], control[top_idx], ancilla)
        idx = top_idx - 1

        for _ in range((idx - 3) // 2):
            toffoli(control[idx - 2], control[idx - 1], control[idx])
            idx -= 2

        toffoli(control[0], control[2], control[3])
        idx = 3

        for _ in range((top_idx - idx) // 2):
            toffoli(control[idx], control[idx + 1], control[idx + 2])
            idx += 2


@guppy
@no_type_check
def _cnx_borrowed_target[n_controls: nat](
    control: array[qubit, n_controls], target: qubit, ancilla: qubit
) -> None:
    n_controls = len(control)
    odd_controls = (n_controls % 2) == 1
    top_idx = n_controls - 1 if odd_controls else n_controls - 2

    for _ in range(2):
        if not odd_controls:
            toffoli(control[top_idx], ancilla, target)
            idx = top_idx - 2

            for _ in range((top_idx - 2) // 2):
                toffoli(control[idx], control[idx + 3], control[idx + 2])
                idx -= 2

            toffoli(control[1], control[3], control[2])
            idx = 4

            for _ in range((top_idx - 2) // 2):
                toffoli(control[idx - 2], control[idx + 1], control[idx])
                idx += 2
        else:
            toffoli(control[n_controls - 1], ancilla, target)
            idx = n_controls - 3

            for _ in range((n_controls - 4) // 2):
                toffoli(control[idx], control[idx + 1], control[idx + 2])
                idx -= 2

            toffoli(control[1], control[3], control[4])
            idx = 4

            for _ in range((n_controls - 4) // 2):
                toffoli(control[idx], control[idx + 1], control[idx + 2])
                idx += 2
