"""Tests for circ utils."""

from guppylang.std.platform import barrier

import pytest
import numpy as np
from typing import no_type_check
from guppylang import guppy
from guppylang.std.builtins import array, output, control
from guppylang.std.quantum import (
    collect_measurements,
    discard,
    measure,
    measure_array,
    qubit,
    x,
    discard_array,
    h,
)
from guppylang.std.debug import state_output
from pytket.circuit import Circuit
from guppyalgos.utils import swap, qarray, transversal, cphase
from guppylang.std.angles import angle
from selene_sim import Quest

from guppyalgos.testing import get_unitary, Endianness, assert_allclose_ignorephase


def test_ccswap_qubits() -> None:
    """Swap two qubits when both controls are set."""

    @guppy
    @no_type_check
    def main() -> None:
        control_0 = qubit()
        control_1 = qubit()
        target_0 = qubit()
        target_1 = qubit()

        x(control_0)
        x(control_1)
        x(target_0)
        with control(control_0, control_1):
            swap(target_0, target_1)

        output("target_0", measure(target_0).read())
        output("target_1", measure(target_1).read())
        discard(control_0)
        discard(control_1)

    result = main.emulator(n_qubits=5).run().collated_shots()[0]
    assert result["target_0"] == [False]
    assert result["target_1"] == [True]


def test_ccswap_registers() -> None:
    """Swap two equal-size registers when both controls are set."""

    @guppy
    @no_type_check
    def main() -> None:
        control_0 = qubit()
        control_1 = qubit()
        target_0 = qarray(2)
        target_1 = qarray(2)

        x(control_0)
        x(control_1)
        x(target_0[0])
        x(target_1[1])
        with control(control_0, control_1):
            swap(target_0, target_1)

        output("target_0", collect_measurements(measure_array(target_0)))
        output("target_1", collect_measurements(measure_array(target_1)))
        discard(control_0)
        discard(control_1)

    result = main.emulator(n_qubits=7).run().results[0].as_dict()
    assert result["target_0"] == [False, True]
    assert result["target_1"] == [True, False]


@pytest.mark.parametrize("theta", [-0.25, 0.33, 0.75])
@no_type_check
def test_cphase(theta: float) -> None:
    """Test the cphase function produces the correct unitary."""
    circ1 = Circuit(2)
    circ1.Rz(theta / 2, 1)  # over rotate control by half
    circ1.CRz(theta, 1, 0)

    circ2 = Circuit(2)
    circ2.CU1(theta, 1, 0)

    # XFAIL here due to global phase difference. Weak condition
    # assert np.allclose(circ1.get_unitary(), circ2.get_unitary(), atol=1e-8)

    # PASS HERE
    assert_allclose_ignorephase(circ1.get_unitary(), circ2.get_unitary())

    @guppy
    def main(state_qreg: array[qubit, 2]) -> None:
        """Guppy program to implement a controlled phase rotation."""
        gtheta = angle(theta)
        cphase(state_qreg[1], state_qreg[0], gtheta)

    guppy_u = get_unitary(main, 2, endianness=Endianness.BIG)

    # XFAIL here due to global phase difference
    assert not np.allclose((circ1.get_unitary()), (guppy_u), atol=1e-8)
    # XFAIL here due to global phase difference
    assert not np.allclose((circ2.get_unitary()), (guppy_u), atol=1e-8)

    assert_allclose_ignorephase(circ1.get_unitary(), guppy_u)
    assert_allclose_ignorephase(circ2.get_unitary(), guppy_u)


def test_cswap_qubit_swaps_when_control_on() -> None:
    """Targets are swapped when the control qubit is |1>."""

    @guppy
    @no_type_check
    def main() -> None:
        c = qubit()
        t1 = qubit()
        t2 = qubit()

        x(c)
        x(t1)
        with control(c):
            swap(t1, t2)

        output("t1", measure(t1).read())
        output("t2", measure(t2).read())
        discard(c)

    result = main.emulator(n_qubits=3).run().collated_shots()[0]
    assert result["t1"] == [False]
    assert result["t2"] == [True]


def test_cswap_register_swaps_when_control_on() -> None:
    """Target registers are swapped when the control qubit is |1>."""

    @guppy
    @no_type_check
    def main() -> None:
        c = qubit()
        t1 = qarray(2)
        t2 = qarray(2)

        x(c)
        x(t1[0])
        x(t2[1])
        with control(c):
            swap(t1, t2)

        output("t1", collect_measurements(measure_array(t1)))
        output("t2", collect_measurements(measure_array(t2)))
        discard(c)

    result = main.emulator(n_qubits=5).run().results[0].as_dict()
    assert result["t1"] == [False, True]
    assert result["t2"] == [True, False]


@pytest.mark.parametrize("n", [1, 2, 3])
def test_cswap_register_no_relative_phase(n: int) -> None:
    """No relative phase is introduced between the control-off/control-on branches."""

    @guppy
    @no_type_check
    def main() -> None:
        c = qubit()
        a = qarray(n)
        b = qarray(n)
        h(c)
        transversal(h, a)
        transversal(h, b)
        with control(c):
            swap(a, b)
        h(c)
        transversal(h, a)
        transversal(h, b)
        barrier(a, b, c)
        state_output("out", a)
        discard_array(a)
        discard_array(b)
        discard(c)

    res = main.emulator(2 * n + 1).run()
    states = Quest.extract_states_dict(res.results[0].entries)

    # total state returned to 0
    assert abs(states["out"].state[0]) == pytest.approx(1)


def test_cswap_qubit_no_relative_phase() -> None:
    """No relative phase is introduced between the control-off/control-on branches."""

    @guppy
    @no_type_check
    def main() -> None:
        c = qubit()
        t1 = qubit()
        t2 = qubit()
        h(c)
        h(t1)
        h(t2)
        with control(c):
            swap(t1, t2)
        h(c)
        h(t1)
        h(t2)
        out = array(c, t1, t2)
        state_output("out", out)
        discard_array(out)

    res = main.emulator(3).run()
    states = Quest.extract_states_dict(res.results[0].entries)

    assert abs(states["out"].state[0]) == pytest.approx(1)
