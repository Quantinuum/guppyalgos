"""Single bit adder and half adders."""

from typing import no_type_check

from guppylang.std.quantum import toffoli, cx, discard_array

from guppylang import guppy, qubit
from guppylang.std.builtins import array, comptime, control, dagger, nat

from guppyalgos.primitives.arithmetic.adder._controls import (
    _acquire_control,
    _release_control,
)
from guppyalgos.primitives.arithmetic.adder.adder_ripple_gidney import (
    _ccx,
    _cccx_toggle,
)
from guppyalgos.utils import qarray


@guppy.unitary
class full_adder:  # noqa: D101
    @guppy(unitary=True)
    @no_type_check
    def __call__(c_in: qubit, a: qubit, b: qubit, c_out: qubit) -> None:
        """Bitwise adder, sum bit computed onto b."""
        half_adder(c_in, a, c_out)
        half_adder(a, b, c_out)
        cx(c_in, a)

    @guppy
    @no_type_check
    def daggered(c_in: qubit, a: qubit, b: qubit, c_out: qubit) -> None:
        """Apply the inverse full adder."""
        with dagger:
            half_adder(c_in, a, c_out)
            half_adder(a, b, c_out)
            cx(c_in, a)

    @guppy
    @no_type_check
    def controlled[m: nat](
        c_in: qubit, a: qubit, b: qubit, c_out: qubit, controls: array[qubit, m]
    ) -> None:
        """Apply the full adder when all controls are one."""
        control_work = qarray(comptime(max(0, m - 1)))
        combined = _acquire_control(controls, control_work)
        with control(combined):
            half_adder(c_in, a, c_out)
            half_adder(a, b, c_out)
        _ccx(combined, c_in, a)
        _release_control(controls, control_work, combined)
        discard_array(control_work)

    @guppy
    @no_type_check
    def ctrl_daggered[m: nat](
        c_in: qubit, a: qubit, b: qubit, c_out: qubit, controls: array[qubit, m]
    ) -> None:
        """Apply the inverse full adder when all controls are one."""
        control_work = qarray(comptime(max(0, m - 1)))
        combined = _acquire_control(controls, control_work)
        _ccx(combined, c_in, a)
        with control(combined), dagger:
            half_adder(c_in, a, c_out)
            half_adder(a, b, c_out)
        _release_control(controls, control_work, combined)
        discard_array(control_work)


@guppy.unitary
class half_adder:  # noqa: D101
    @guppy(unitary=True)
    @no_type_check
    def __call__(a: qubit, b: qubit, c_out: qubit) -> None:
        """Bitwise half-adder, sum bit computed onto b."""
        toffoli(a, b, c_out)
        cx(a, b)

    @guppy
    @no_type_check
    def daggered(a: qubit, b: qubit, c_out: qubit) -> None:
        """Apply the inverse half adder."""
        cx(a, b)
        toffoli(a, b, c_out)

    @guppy
    @no_type_check
    def controlled[m: nat](
        a: qubit, b: qubit, c_out: qubit, controls: array[qubit, m]
    ) -> None:
        """Apply the half adder when all controls are one."""
        control_work = qarray(comptime(max(0, m - 1)))
        combined = _acquire_control(controls, control_work)
        _cccx_toggle(combined, a, b, c_out)
        _ccx(combined, a, b)
        _release_control(controls, control_work, combined)
        discard_array(control_work)

    @guppy
    @no_type_check
    def ctrl_daggered[m: nat](
        a: qubit, b: qubit, c_out: qubit, controls: array[qubit, m]
    ) -> None:
        """Apply the inverse half adder when all controls are one."""
        control_work = qarray(comptime(max(0, m - 1)))
        combined = _acquire_control(controls, control_work)
        _ccx(combined, a, b)
        _cccx_toggle(combined, a, b, c_out)
        _release_control(controls, control_work, combined)
        discard_array(control_work)
