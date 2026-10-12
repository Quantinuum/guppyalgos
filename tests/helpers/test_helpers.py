"""Test testing helpers."""

import itertools
from typing import cast, no_type_check

import numpy as np
import pytest
from guppylang.decorator import guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.emulator._args import EntrypointArgValueError
from guppylang.std.angles import angle
from guppylang.std.builtins import array, output
from guppylang.std.debug import state_output
from guppylang.std.platform import barrier
from guppylang.std.quantum import (
    cx,
    discard,
    discard_array,
    h,
    qubit,
    ry,
    toffoli,
    x,
    cy,
)
from hugr.qsystem.result import QsysResult
from numpy.typing import NDArray
from selene_quest_plugin import SeleneQuestState
from selene_sim import Quest, build

from guppyalgos.utils import qarray
from guppyalgos.testing import (
    SubQuestState,
    _statevector_with_zeroed_ancilla,
    assert_allclose_ignorephase,
    assert_cntrl_unitary,
    extract_state_branches_in_superposition,
    get_statevector,
    get_statevector_projected,
    get_unitary,
    get_unitary_assumed_phase,
    get_unitary_projected,
    project_state_onto_bitstring,
    switch_matrix_endianness,
    switch_endianness,
)


def _entangled_3reg_main() -> tuple[GuppyFunctionDefinition, int]:
    """Build a shared three-register entangled example for projection tests.

    Returns:
        A tuple containing the guppy ``main`` function, and the total qubit count.

    """
    n_state_qubits = 2
    n_phase_qubits = 3
    n_total_qubits = n_state_qubits + n_phase_qubits + 1
    state_preparation_angle = np.pi / 4

    @guppy
    @no_type_check
    def main() -> None:
        phase_qreg_arr = qarray(n_phase_qubits)
        state_qreg_arr = qarray(n_state_qubits)
        ancilla_qreg_arr = qarray(1)

        ry(state_qreg_arr[0], angle(state_preparation_angle))
        x(state_qreg_arr[1])

        for i in range(n_phase_qubits):
            h(phase_qreg_arr[i])

        h(ancilla_qreg_arr[0])

        for i in range(n_phase_qubits):
            cx(phase_qreg_arr[i], ancilla_qreg_arr[0])
            cx(phase_qreg_arr[i], state_qreg_arr[i % n_state_qubits])

        for i in range(n_state_qubits):
            cx(state_qreg_arr[i], ancilla_qreg_arr[0])

        state_output("phase", phase_qreg_arr)
        state_output("ancilla", ancilla_qreg_arr)
        state_output("system", state_qreg_arr)
        discard_array(phase_qreg_arr)
        discard_array(state_qreg_arr)
        discard_array(ancilla_qreg_arr)

    return main, n_total_qubits


def _build_unstructured_multi_register_program(
    order_option: int,
) -> tuple[GuppyFunctionDefinition[[], None], int]:
    """Build a guppy program with several registers of different sizes.

    Order option in [0,1,2] allocs the registers in different orders to test that
    projections do not depend on this.
    Act with gates to produce a slightly entangled state with unstructured coefs.
    """

    @guppy
    @no_type_check
    def main() -> None:
        if order_option == 0:
            small = qarray(1)
            medium = qarray(2)
            large = qarray(3)
        elif order_option == 1:
            large = qarray(3)
            small = qarray(1)
            medium = qarray(2)
        else:
            medium = qarray(2)
            large = qarray(3)
            small = qarray(1)

        ry(small[0], angle(0.45))
        ry(medium[0], angle(0.63))
        ry(medium[1], angle(0.78))
        ry(large[0], angle(0.34))
        ry(large[2], angle(0.91))

        toffoli(small[0], medium[0], large[1])
        cx(medium[1], large[2])
        cx(large[0], medium[0])
        ry(large[1], angle(0.52))

        barrier(small, medium, large)
        state_output("small", small)
        state_output("medium", medium)
        state_output("large", large)
        discard_array(small)
        discard_array(medium)
        discard_array(large)

    total_qubits = 1 + 2 + 3
    return main, total_qubits


def _project_registers_in_order(
    state: SeleneQuestState,
    specified_qubits: dict[str, list[int]],
    projection_order: list[str],
    bitstrings: dict[str, list[bool]],
) -> SubQuestState:
    projected: list[str] = []
    current = state

    for projected_register in projection_order:
        current.specified_qubits = specified_qubits[projected_register]
        projected.append(projected_register)
        current = project_state_onto_bitstring(
            current,
            bitstrings[projected_register],
            new_specified_qubits=[],
        ).state

    return cast("SubQuestState", current)


def test_project_state_onto_bitstring() -> None:
    """Test correct behavior for known examples and 0 norm case.

    TODO add more examples
    """

    @guppy
    def main() -> None:
        a = qarray(2)
        q = qarray(2)
        x(a[0])
        h(a[0])
        cx(a[0], q[0])
        state_output("out", a)
        state_output("desired_idx", q)
        discard_array(q)
        discard_array(a)

    HUGR = main.compile()
    runner = build(HUGR)

    shots = QsysResult(
        runner.run_shots(
            simulator=Quest(),
            n_qubits=4,
            n_shots=1,
        )
    )

    for shot in shots.results:
        states = Quest.extract_states_dict(shot.entries)
        desired_qubits = states["desired_idx"].specified_qubits
        # assert here will fail if state_outputs get moved through gates
        assert_allclose_ignorephase(states["desired_idx"].state, states["out"].state)
        res01 = project_state_onto_bitstring(
            states["out"], [True, False], new_specified_qubits=desired_qubits
        )
        np.testing.assert_allclose(res01.probability, 1 / 2)
        # state normalized
        np.testing.assert_allclose(np.linalg.norm(res01.state.state), 1)

        res11 = project_state_onto_bitstring(
            states["out"], [True, True], new_specified_qubits=desired_qubits
        )
        np.testing.assert_allclose(res11.probability, 0)

        # test state is correct TODO needs global phase
        assert_allclose_ignorephase(res01.state.state, np.array([0, 1, 0, 0]))


def _expected_ry_output_state(
    control: bool, angle_half_turns: float
) -> NDArray[np.complex128]:
    """Return the Ry matrix column selected by the control value."""
    half_angle = np.pi * angle_half_turns / 2
    ry_matrix = np.array(
        [
            [np.cos(half_angle), -np.sin(half_angle)],
            [np.sin(half_angle), np.cos(half_angle)],
        ],
        dtype=np.complex128,
    )
    return ry_matrix[:, int(control)]


def _expected_ry_probability(control: bool, angle_half_turns: float) -> float:
    """Return the analytical probability of an Ry-prepared control value."""
    half_angle = np.pi * angle_half_turns / 2
    amplitude = np.sin(half_angle) if control else np.cos(half_angle)
    return float(amplitude**2)


def test_extract_state_branches_in_superposition() -> None:
    """Extract every branch and retain only the requested result register."""
    branch_ry_angle = 1 / 3
    result_ry_angle = 1 / 5

    @guppy
    @no_type_check
    def main() -> None:
        branch = qubit()
        result = qubit()
        ry(branch, angle(branch_ry_angle))
        cx(branch, result)
        ry(result, angle(result_ry_angle))
        barrier(branch, result)
        state_output("branch", branch)
        state_output("result", result)
        discard(branch)
        discard(result)

    shot = main.emulator(2).run().results[0]
    states = Quest.extract_states_dict(shot.entries)
    branches = extract_state_branches_in_superposition(
        states,
        branch_tag="branch",
        result_tags=["result"],
        branch_bitstrings=[[False], [True]],
    )

    assert set(branches) == {(False,), (True,)}
    for control in [False, True]:
        np.testing.assert_allclose(
            branches[(control,)].probability,
            _expected_ry_probability(control, branch_ry_angle),
        )
        assert_allclose_ignorephase(
            branches[(control,)].state.state,
            _expected_ry_output_state(control, result_ry_angle),
        )


def test_extract_state_branches_uses_little_endian_bitstrings() -> None:
    """Interpret multi-qubit branch bitstrings with the least-significant bit first."""

    @guppy
    @no_type_check
    def main() -> None:
        branch = qarray(2)
        result = qubit()
        x(branch[0])
        state_output("branch", branch)
        state_output("result", result)
        discard_array(branch)
        discard(result)

    shot = main.emulator(3).run().results[0]
    states = Quest.extract_states_dict(shot.entries)
    branch_bits_le = [True, False]
    branches = extract_state_branches_in_superposition(
        states,
        branch_tag="branch",
        result_tags=["result"],
        branch_bitstrings=[branch_bits_le],
    )

    projected = branches[tuple(branch_bits_le)]
    np.testing.assert_allclose(projected.probability, 1.0)
    assert_allclose_ignorephase(
        projected.state.state,
        np.array([1.0, 0.0], dtype=np.complex128),
    )


def test_project_state_wrong_bitstring_length() -> None:
    """Error on incorrect bitstring lengths."""

    @guppy
    @no_type_check
    def main() -> None:
        q = qarray(2)
        state_output("out", q)
        discard_array(q)

    HUGR = main.compile()
    runner = build(HUGR)

    shots = QsysResult(
        runner.run_shots(
            simulator=Quest(),
            n_qubits=2,
            n_shots=1,
        )
    )

    for shot in shots.results:
        states = Quest.extract_states_dict(shot.entries)
        with pytest.raises(ValueError, match="bitstring") as _:
            project_state_onto_bitstring(states["out"], [False])
        with pytest.raises(ValueError, match="bitstring") as _:
            project_state_onto_bitstring(states["out"], [False, True, True])


def test_project_state_nondisjoint_specified_qubits() -> None:
    """Error if the projection qubits overlap with the new specified ones."""

    @guppy
    @no_type_check
    def main() -> None:
        q = qarray(2)
        state_output("out", q)
        discard_array(q)

    HUGR = main.compile()
    runner = build(HUGR)

    shots = QsysResult(
        runner.run_shots(
            simulator=Quest(),
            n_qubits=2,
            n_shots=1,
        )
    )

    for shot in shots.results:
        states = Quest.extract_states_dict(shot.entries)

        # check that using new_specified_qubits=None works
        projected_state = project_state_onto_bitstring(
            states["out"], [False, True], new_specified_qubits=None
        )
        assert len(projected_state.state.specified_qubits) == 0

        # error if new_specified_qubits are part of the old ones
        with pytest.raises(ValueError, match="overlap") as _:
            project_state_onto_bitstring(
                states["out"], [False, True], new_specified_qubits=[0]
            )

        # error if the new indices are not in the correct range
        with pytest.raises(ValueError, match="range") as _:
            project_state_onto_bitstring(
                states["out"], [False, True], new_specified_qubits=[-1]
            )
        with pytest.raises(ValueError, match="range") as _:
            project_state_onto_bitstring(
                states["out"], [False, True], new_specified_qubits=[2]
            )


def test_get_unitary() -> None:
    """Check that get_unitary returns the expected unitary."""

    # test cx
    @guppy
    @no_type_check
    def main(qreg: array[qubit, 2]) -> None:
        cx(qreg[0], qreg[1])

    expected = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ],
        dtype=np.complex128,
    )

    actual = get_unitary(main, 2)

    assert_allclose_ignorephase(actual, expected)

    # test single qubit rotation
    @guppy
    @no_type_check
    def main(qreg: array[qubit, 1]) -> None:
        ry(qreg[0], angle(0.5))

    half_turn_angle = np.pi / 4
    expected = np.array(
        [
            [np.cos(half_turn_angle), -np.sin(half_turn_angle)],
            [np.sin(half_turn_angle), np.cos(half_turn_angle)],
        ],
        dtype=np.complex128,
    )

    actual = get_unitary(main, 1)

    assert_allclose_ignorephase(actual, expected)


@pytest.mark.parametrize("unitary_helper", [get_unitary, get_unitary_assumed_phase])
def test_get_unitary_with_zeroed_internal_ancilla(unitary_helper) -> None:
    """Extract only the logical register when an internal ancilla returns to zero."""

    @guppy
    @no_type_check
    def main(state: array[qubit, 1]) -> None:
        ancilla = qubit()
        cx(state[0], ancilla)
        cx(state[0], ancilla)
        x(state[0])
        discard(ancilla)

    actual = unitary_helper(main, 1, n_extra_qubits=1)

    assert_allclose_ignorephase(actual, np.array([[0, 1], [1, 0]]))


@pytest.mark.parametrize("unitary_helper", [get_unitary, get_unitary_assumed_phase])
def test_get_unitary_rejects_nonzero_internal_ancilla(unitary_helper) -> None:
    """Reject a unitary whose internal ancilla is not returned to zero."""

    @guppy
    @no_type_check
    def main(state: array[qubit, 1]) -> None:
        ancilla = qubit()
        x(ancilla)
        discard(ancilla)

    with pytest.raises(
        ValueError,
        match="Internal ancilla qubits were not returned to the zero state",
    ):
        unitary_helper(main, 1, n_extra_qubits=1)


@pytest.mark.parametrize(
    ("pre_select", "post_select", "expected"),
    [
        ([False], [False], np.eye(2)),
        ([True], [True], np.array([[0, -1j], [1j, 0]])),
    ],
)
def test_get_unitary_projected_cntrl_y(
    pre_select: list[bool],
    post_select: list[bool],
    expected: NDArray[np.complex128],
) -> None:
    """Check projected blocks of a controlled-Y circuit."""

    @guppy
    @no_type_check
    def projected_main(
        projection: array[qubit, 1],
        state: array[qubit, 1],
    ) -> None:
        cy(projection[0], state[0])
        state_output("projection", projection)

    projected = get_unitary_projected(
        projected_main,
        1,
        post_select_dict={"projection": post_select},
        pre_select_dict={"projection": pre_select},
    )

    assert_allclose_ignorephase(projected, expected)


def test_assert_cntrl_unitary() -> None:
    """Check the coherent blocks of a controlled-Y circuit."""

    @guppy
    @no_type_check
    def projected_main(
        control: array[qubit, 1],
        state: array[qubit, 1],
    ) -> None:
        h(control[0])
        cy(control[0], state[0])
        h(control[0])

    expected_y = np.array([[0, -1j], [1j, 0]])
    assert_cntrl_unitary(projected_main, expected_y, 1, {})


def test_get_unitary_projected_with_zeroed_internal_ancilla() -> None:
    """Extract only the state register when an internal ancilla returns to zero."""

    @guppy
    @no_type_check
    def projected_main(
        projection: array[qubit, 1],
        state: array[qubit, 1],
    ) -> None:
        ancilla = qubit()
        cx(state[0], ancilla)
        cx(state[0], ancilla)
        x(state[0])
        discard(ancilla)

    projected = get_unitary_projected(
        projected_main,
        1,
        post_select_dict={"projection": [False]},
        n_extra_qubits=1,
    )

    assert_allclose_ignorephase(projected, np.array([[0, 1], [1, 0]]))


def test_statevector_with_zeroed_ancilla_preserves_state() -> None:
    """Remove zeroed ancilla without renormalizing or changing phase."""
    expected = np.array([0.25j, -0.5], dtype=np.complex128)
    state = SeleneQuestState(
        state=np.array([expected[0], expected[1], 0.0, 0.0]),
        total_qubits=2,
        specified_qubits=[0],
    )

    actual = _statevector_with_zeroed_ancilla(state)

    np.testing.assert_allclose(actual, expected)


def test_get_unitary_projected_rejects_nonzero_internal_ancilla() -> None:
    """Reject a projected unitary whose internal ancilla is not returned to zero."""

    @guppy
    @no_type_check
    def projected_main(
        projection: array[qubit, 1],
        state: array[qubit, 1],
    ) -> None:
        ancilla = qubit()
        x(ancilla)
        discard(ancilla)

    with pytest.raises(
        ValueError,
        match="Internal ancilla qubits were not returned to the zero state",
    ):
        get_unitary_projected(
            projected_main,
            1,
            post_select_dict={"projection": [False]},
            n_extra_qubits=1,
        )


def test_unitary_projected_ordering() -> None:
    """Check test_unitary_projected uses correct ordering on projection bits."""

    @guppy
    @no_type_check
    def projected_main(
        projection: array[qubit, 2],
        state: array[qubit, 1],
    ) -> None:
        x(projection[0])
        toffoli(projection[0], projection[1], state[0])
        x(projection[0])

    # bits where control fires
    post_dict_x = {"anc": [False, True]}
    pre_dict_x = {"anc": [False, True]}

    # reverse
    post_dict_I = {"anc": [True, False]}
    pre_dict_I = {"anc": [True, False]}

    should_be_x_matrix = get_unitary_projected(
        projected_main, 1, pre_select_dict=pre_dict_x, post_select_dict=post_dict_x
    )
    should_be_I_matrix = get_unitary_projected(
        projected_main, 1, pre_select_dict=pre_dict_I, post_select_dict=post_dict_I
    )
    assert_allclose_ignorephase(should_be_x_matrix, np.array([[0, 1], [1, 0]]))
    assert_allclose_ignorephase(should_be_I_matrix, np.eye(2))


def test_get_statevector_projected_3reg() -> None:
    """Test get_statevector_projected on two registers."""

    @guppy
    @no_type_check
    def main() -> None:
        a = qubit()
        b = qubit()
        c = qubit()

        x(a)
        x(b)
        toffoli(a, b, c)
        state_output("a", a)
        state_output("b", b)

        discard(a)
        discard(b)
        discard(c)

    expected = np.array([0, 1])
    result = get_statevector_projected(
        main,
        3,
        post_select_dict={"a": [True], "b": [True]},
    )

    assert_allclose_ignorephase(result, expected)


def test_get_statevector_projected_rejects_unknown_register() -> None:
    """Unknown state_result tags should raise a ValueError."""
    main, n_total_qubits = _entangled_3reg_main()

    with pytest.raises(ValueError, match="subset of the available state keys"):
        get_statevector_projected(
            main,
            n_total_qubits,
            {"not_a_register": [True]},
        )


def test_get_statevector_projected_rejects_wrong_bitstring_length() -> None:
    """Post-selection bitstrings must match the projected register width."""
    main, n_total_qubits = _entangled_3reg_main()

    with pytest.raises(ValueError, match="specified qubits"):
        get_statevector_projected(
            main,
            n_total_qubits,
            {"phase": [True, False]},
        )


def test_project_state_same_for_different_projection_orders() -> None:
    """Projecting on disjoint registers should not depend on the order."""
    main, total_qubits = _build_unstructured_multi_register_program(0)
    shots = main.emulator(total_qubits).run()

    for shot in shots.results:
        states = Quest.extract_states_dict(shot.entries)
        spec_dict = {
            key: states[key].specified_qubits for key in ["small", "medium", "large"]
        }

        bitstrings = {
            "small": [False],
            "medium": [True, False],
            "large": [False, True, True],
        }

        for remaining in ["small", "medium", "large"]:
            projected_registers = [name for name in spec_dict if name != remaining]
            canonical = None
            print(projected_registers)
            for projection_order in itertools.permutations(projected_registers):
                result = _project_registers_in_order(
                    states["small"],  # one of them doesn't matter which
                    spec_dict,
                    list(projection_order),
                    bitstrings,
                )

                if canonical is None:
                    canonical = result
                    continue

                assert_allclose_ignorephase(result.state, canonical.state)


def test_project_state_same_for_different_array_initialization_orders() -> None:
    """Register initialization order must not change projection outcomes."""
    order_options = [0, 1, 2]

    bitstrings = {
        "small": [False],
        "medium": [True, False],
        "large": [False, True, True],
    }

    projected_results: list[SubQuestState] = []

    for order_option in order_options:
        main, total_qubits = _build_unstructured_multi_register_program(order_option)
        shots = main.emulator(total_qubits).run()
        shot = shots.results[0]
        states = Quest.extract_states_dict(shot.entries)
        spec_dict = {
            key: states[key].specified_qubits for key in ["small", "medium", "large"]
        }

        projected_results.append(
            _project_registers_in_order(
                states["small"],  # one of them doesn't matter which
                spec_dict,
                ["small", "medium"],
                bitstrings,
            )
        )

    baseline = projected_results[0]
    for result in projected_results[1:]:
        assert_allclose_ignorephase(result.state, baseline.state)


def test_passing_precompiled_through_helpers() -> None:
    """Test that either guppy, emulator, or hugr can be passed to helpers."""
    n = 1

    @guppy
    @no_type_check
    def main() -> None:
        q = qubit()
        state_output("result_state", q)
        discard(q)

    guppy_res = get_statevector(main, n)
    package_res = get_statevector(main.compile(), n)
    emulator_res = get_statevector(main.emulator(n), n)
    np.testing.assert_allclose(guppy_res, package_res)
    np.testing.assert_allclose(package_res, emulator_res)
    # ----------
    guppy_res = get_statevector_projected(main, n, {"result_state": [False]})
    package_res = get_statevector_projected(
        main.compile(), n, {"result_state": [False]}
    )
    emulator_res = get_statevector_projected(
        main.emulator(n), n, {"result_state": [False]}
    )
    np.testing.assert_allclose(guppy_res, package_res)
    np.testing.assert_allclose(package_res, emulator_res)


def test_helper_entrypoint_args() -> None:
    """Test entrypoint args work in helpers."""

    @guppy
    @no_type_check
    def main(a: int) -> None:
        q = qubit()
        state_output("result_state", q)
        output("argument", a)
        discard(q)

    arg = 3
    res = main.emulator(1).run(a=3).results[0]
    arg_output = res.as_dict()["argument"]
    assert arg == arg_output

    get_statevector(main, 1, main_args_dict={"a": 1})
    with pytest.raises(EntrypointArgValueError):
        get_statevector(main, 1)


def test_switch_matrix_endianness() -> None:
    """Tests matrix endianness is switched correctly."""
    input_matrix = np.array(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
            [12, 13, 14, 15],
        ]
    )
    desired_output = np.array(
        [
            [0, 2, 1, 3],
            [8, 10, 9, 11],
            [4, 6, 5, 7],
            [12, 14, 13, 15],
        ]
    )
    assert np.all(switch_matrix_endianness(input_matrix) == desired_output)


def test_subqueststate() -> None:
    """Test all methods of SubQuestState act correctly on local qubits.

    And that interface is the same as SeleneQuestState.
    """
    n_qubits = 2
    state = np.zeros(2**n_qubits, dtype=np.complex128)
    # set bits to [1, 0] (little endian)
    state[1] = 1
    # have them come from a larger state with 4 qubits
    # with the ones we care about being the first and last
    substate = SubQuestState(
        state=state,
        total_qubits=n_qubits,
        specified_qubits=[0, 3],
        projected_out_qubits=[1, 2],
    )

    np.testing.assert_array_equal(substate.state, state)
    assert substate.specified_qubits == [0, 3]
    assert substate._reindex_global_to_local([0, 3]) == [0, 1]
    assert substate._reindex_local_to_global([0, 1]) == [0, 3]

    # test that methods behave in the same way as SeleneQuestState
    # .get_single_state methods etc return big endian vectors
    # as opposed to the internal little endian .state
    # (this is the same as SeleneQuestState behavior)
    expected_subsystem_state_be = switch_endianness(state)
    np.testing.assert_array_equal(
        substate.get_density_matrix(),
        np.outer(expected_subsystem_state_be, expected_subsystem_state_be),
    )
    distribution = substate.get_state_vector_distribution()
    assert len(distribution) == 1
    np.testing.assert_array_equal(distribution[0].state, expected_subsystem_state_be)
    assert distribution[0].probability == 1
    np.testing.assert_array_equal(
        substate.get_single_state(), expected_subsystem_state_be
    )
    assert str(substate.get_dirac_notation()[0].state) == "|10>"
    assert str(substate.get_single_dirac_notation()) == "|10>"


def test_subqueststate_rejects_invalid_indices() -> None:
    """Test invalid setting of subqueststate specified qubits."""
    state = SubQuestState(
        np.array([1, 0, 0, 0], dtype=np.complex128),
        total_qubits=2,
        specified_qubits=[0, 3],
        projected_out_qubits=[1, 2],
    )

    # cant specify an already projected out qubit
    with pytest.raises(ValueError, match="overlap"):
        state.specified_qubits = [1]

    # qubit out of range of total
    with pytest.raises(ValueError, match="range"):
        state.specified_qubits = [4]


def test_subqueststate_single_state_rejects_entangled_subsystem() -> None:
    """Single-state getters reject a subsystem entangled with an unspecified qubit."""
    # The full Bell state is pure, but specifying only qubit 0 gives a mixed
    # reduced state
    bell_state = np.array([1, 0, 0, 1], dtype=np.complex128) / np.sqrt(2)
    substate = SubQuestState(
        state=bell_state,
        total_qubits=2,
        specified_qubits=[0],
        projected_out_qubits=[],
    )

    # Neither getter can return one pure state for this mixed subsystem.
    with pytest.raises(ValueError, match="not a pure state"):
        substate.get_single_state()
    with pytest.raises(ValueError, match="not a pure state"):
        substate.get_single_dirac_notation()
