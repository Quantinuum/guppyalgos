"""Tests for circ utils."""

from typing import no_type_check

import numpy as np
import pytest
from guppylang import guppy
from guppylang.std.angles import angle
from guppylang.std.builtins import array, control, dagger, output
from guppylang.std.debug import state_output
from guppylang.std.quantum import (
    collect_measurements,
    cx,
    discard,
    discard_array,
    h,
    measure,
    measure_array,
    qubit,
    rz,
    t,
    x,
)
from pytket.circuit import Circuit
from selene_sim import Quest

from guppyalgos.primitives.utilities import t_state
from guppyalgos.primitives.utilities.gates import ccswap, cphase, transversal
from guppyalgos.testing import (
    Endianness,
    assert_allclose_ignorephase,
    assert_unitary_modifiers,
    get_unitary,
)
from guppyalgos.utils import qarray


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
        ccswap(control_0, control_1, target_0, target_1)

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
        ccswap(control_0, control_1, target_0, target_1)

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


@pytest.mark.parametrize("n_ctrl", [1, 2, 3])
def test_controlled_transversal_shared_work_qubit(n_ctrl: int) -> None:
    """Controlled transversal ops match per-qubit controlled ops and invert."""
    n = 3

    @guppy
    @no_type_check
    def prepare(
        cs: array[qubit, n_ctrl], a: array[qubit, n], b: array[qubit, n]
    ) -> None:
        transversal(h, cs)
        transversal(h, a)
        transversal(h, b)
        for i in range(n):
            rz(a[i], angle(0.3 * (i + 1)))
            rz(b[i], angle(0.7 * (i + 1)))

    @guppy
    @no_type_check
    def unprepare(
        cs: array[qubit, n_ctrl], a: array[qubit, n], b: array[qubit, n]
    ) -> None:
        for i in range(n):
            rz(a[i], angle(-0.3 * (i + 1)))
            rz(b[i], angle(-0.7 * (i + 1)))
        transversal(h, cs)
        transversal(h, a)
        transversal(h, b)

    @guppy
    @no_type_check
    def matches_pairwise() -> None:
        cs = qarray(n_ctrl)
        a = qarray(n)
        b = qarray(n)
        prepare(cs, a, b)
        with control(cs):
            transversal(cx, a, b, array(1))
            transversal(t, a, 0, 2)
            transversal(cx, a, b, 1, 3)
        # undo per qubit: the identity only if the two implementations agree
        for i in range(1, n):
            with control(cs), dagger:
                cx(a[i], b[i])
        with control(cs), dagger:
            t(a[1])
            t(a[0])
        with control(cs), dagger:
            cx(a[0], b[0])
            cx(a[2], b[2])
        unprepare(cs, a, b)
        state_output("out", a)
        discard_array(cs)
        discard_array(a)
        discard_array(b)

    @guppy
    @no_type_check
    def roundtrip() -> None:
        cs = qarray(n_ctrl)
        a = qarray(n)
        b = qarray(n)
        prepare(cs, a, b)
        with control(cs):
            transversal(cx, a, b, array(1))
            transversal(cx, a, b, 1, 3)
        with control(cs), dagger:
            transversal(cx, a, b, 1, 3)
            transversal(cx, a, b, array(1))
        unprepare(cs, a, b)
        state_output("out", a)
        discard_array(cs)
        discard_array(a)
        discard_array(b)

    for prog in (matches_pairwise, roundtrip):
        res = prog.emulator(n_ctrl + 2 * n + 2).run()
        state = Quest.extract_states_dict(res.results[0].entries)["out"].state
        # |0..0> also shows the shared work qubit was uncomputed
        assert abs(state[0]) == pytest.approx(1)


def test_t_state_modifiers() -> None:
    """Check H followed by T and its modifiers on arbitrary inputs."""

    @guppy(unitary=True)
    @no_type_check
    def operation(qs: array[qubit, 1]) -> None:
        t_state(qs[0])

    hadamard = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    expected = np.diag([1, np.exp(0.25j * np.pi)]) @ hadamard
    assert_unitary_modifiers(operation, 1, expected_unitary=expected)
