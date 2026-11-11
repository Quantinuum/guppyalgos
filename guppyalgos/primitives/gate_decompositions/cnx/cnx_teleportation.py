"""Multicontrolled x gate based on teleportation."""

from math import ceil, log2
from typing import no_type_check

from guppylang.decorator import guppy
from guppylang.std.builtins import array, comptime, nat, owned
from guppylang.std.quantum import (
    collect_measurements,
    cx,
    cz,
    h,
    measure_array,
    qubit,
    toffoli,
    x,
)

from guppyalgos.primitives.gate_decompositions.cnx.cnx import unitary_modify_cnx
from guppyalgos.primitives.utilities.gates import transversal
from guppyalgos.utils import qarray


@unitary_modify_cnx
@guppy
@no_type_check
def cnx_teleportation[n_controls: nat](
    control: array[qubit, n_controls], target: qubit
) -> None:
    r"""Apply cnx using the teleportation protocol.

    Based on https://arxiv.org/abs/2604.25861, requires O(n) ancillary qubits and a
    linear Toffoli count with a constant depth.

    Args:
        control (array[qubit, n_controls]): register of control qubits.
        target (qubit): target qubit.

    """
    if n_controls == 1:
        cx(control[0], target)
    elif n_controls == 2:
        toffoli(control[0], control[1], target)
    else:
        _cnx_teleportation_aux_l0(control, target)


@guppy.comptime
@no_type_check
def _get_mi(n: nat @ comptime) -> int:
    """Compute the number of control pairs."""
    return int((2 * (n) - 1 + (-1) ** (n)) // 4)


@guppy.comptime
@no_type_check
def _get_li(n: nat @ comptime) -> int:
    """Compute the number of left controls."""
    return int((1 - (-1) ** n) // 2)


@guppy.comptime
@no_type_check
def _get_mi_extra(n: nat @ comptime) -> int:
    """Compute the number of control pairs with an extra qubit."""
    return int((2 * (n + 1) - 1 + (-1) ** (n + 1)) // 4)


@guppy.comptime
@no_type_check
def _get_li_extra(n: nat @ comptime) -> int:
    """Compute the number of left controls with an extra qubit."""
    return int((1 - (-1) ** (n + 1)) // 2)


@guppy
@no_type_check
def _pre_cnx[n_controls: nat, n_ancillas: nat](
    control: array[qubit, n_controls],
    ancillas_z: array[qubit, n_ancillas],
    ancillas_x: array[qubit, n_ancillas],
) -> None:
    """Create Bell pairs for ancillas and apply first round of toffolis."""
    transversal(h, ancillas_z)
    transversal(cx, ancillas_z, ancillas_x)
    # ancillas z
    for i in range(_get_mi(n_controls)):
        q1_index = 2 * i
        q2_index = q1_index + 1
        toffoli(control[q1_index], control[q2_index], ancillas_z[i])


@guppy
@no_type_check
def _measure_and_conditional_z[n_ancillas: nat](
    ancillas_z: array[qubit, n_ancillas] @ owned,  # ty: ignore[not-subscriptable]
    ancillas_x: array[qubit, n_ancillas],
) -> None:
    """Measure z ancillas and apply conditional gates on x ancillas."""
    z_meas = collect_measurements(measure_array(ancillas_z))
    for i in range(n_ancillas):
        if z_meas[i]:
            x(ancillas_x[i])


@guppy
@no_type_check
def _post_cnx[n_controls: nat, n_ancillas: nat](
    control: array[qubit, n_controls],
    ancillas_x: array[qubit, n_ancillas] @ owned,  # ty: ignore[not-subscriptable]
) -> array[bool, n_ancillas]:
    """Measure x ancillas on x basis and apply conditional gates on controls."""
    transversal(h, ancillas_x)
    x_meas = collect_measurements(measure_array(ancillas_x))

    # controlled zs
    for i in range(_get_mi(n_controls)):
        q1_index = 2 * i
        q2_index = q1_index + 1

        if x_meas[i]:
            cz(control[q1_index], control[q2_index])
    return x_meas


@guppy
@no_type_check
def _cnx_teleportation_aux_l0[n_controls: nat](
    control: array[qubit, n_controls], target: qubit
) -> None:
    """Recursive function that applies that implements a cnx with telep for l=0."""
    ancillas_z = qarray(comptime(int(2 * n_controls - 1 + (-1) ** n_controls) // 4))
    ancillas_x = qarray(comptime(int(2 * n_controls - 1 + (-1) ** n_controls) // 4))
    _pre_cnx(control, ancillas_z, ancillas_x)
    _measure_and_conditional_z(ancillas_z, ancillas_x)

    # cnx
    if _get_mi(n_controls) + _get_li(n_controls) == 2:
        if _get_li(n_controls) == 1:
            toffoli(ancillas_x[0], control[comptime(n_controls - 1)], target)
        else:
            toffoli(ancillas_x[0], ancillas_x[1], target)

    else:
        if _get_li(n_controls) == 1:
            _cnx_teleportation_aux_l1(
                ancillas_x, control[comptime(n_controls - 1)], target
            )
        else:
            _cnx_teleportation_aux_l0(ancillas_x, target)
    _post_cnx(control, ancillas_x)


@guppy
@no_type_check
def _cnx_teleportation_aux_l1[n_controls: nat](
    control: array[qubit, n_controls], extra_control: qubit, target: qubit
) -> None:
    """Recursive function that applies that implements a cnx with telep for l=1."""
    ancillas_z = qarray(
        comptime(int(2 * (n_controls + 1) - 1 + (-1) ** (n_controls + 1)) // 4)
    )
    ancillas_x = qarray(
        comptime(int(2 * (n_controls + 1) - 1 + (-1) ** (n_controls + 1)) // 4)
    )

    _pre_cnx(control, ancillas_z, ancillas_x)

    if _get_li_extra(n_controls) == 0:
        toffoli(
            control[comptime(n_controls - 1)],
            extra_control,
            ancillas_z[_get_mi_extra(n_controls) - 1],
        )

    _measure_and_conditional_z(ancillas_z, ancillas_x)

    # cnx
    if _get_mi_extra(n_controls) + _get_li_extra(n_controls) == 2:
        if _get_li_extra(n_controls) == 1:
            toffoli(ancillas_x[0], extra_control, target)
        else:
            toffoli(ancillas_x[0], ancillas_x[1], target)

    else:
        if _get_li_extra(n_controls) == 1:
            _cnx_teleportation_aux_l1(ancillas_x, extra_control, target)
        else:
            _cnx_teleportation_aux_l0(ancillas_x, target)

    x_meas = _post_cnx(control, ancillas_x)
    if _get_li_extra(n_controls) == 0:
        if x_meas[_get_mi_extra(n_controls) - 1]:
            cz(control[comptime(n_controls - 1)], extra_control)


def _get_num_ancillas_cnx_teleportation(n_controls: int) -> int:
    """Compute the number of ancillas required for the telep cnx."""
    n_ancillas = 0
    i_max = 1 if n_controls == 2 else (ceil(log2(n_controls)) - 1)

    n_i = n_controls
    for _ in range(i_max):
        m_i = (2 * n_i - 1 + (-1) ** n_i) // 4
        l_i = (1 - (-1) ** n_i) // 2
        n_ancillas += 2 * m_i
        n_i = m_i + l_i

    return n_ancillas
