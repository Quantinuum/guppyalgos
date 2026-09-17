"""Helpers for inspecting Guppy programs with a statevector simulator."""

from __future__ import annotations
from guppylang.std.platform import barrier

from enum import IntEnum
from itertools import chain
from typing import Any, no_type_check

import numpy as np
from numpy.polynomial.chebyshev import chebval
from guppylang import guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.emulator import EmulatorInstance
from guppylang.std.builtins import array, comptime, nat
from guppylang.std.debug import state_output
from guppylang.std.quantum import discard_array, qubit
from hugr.package import Package
from hugr.qsystem.result import QsysResult
from numpy.typing import NDArray
from selene_quest_plugin.state import SeleneQuestState, TracedState
from selene_sim import Quest, build

from guppyalgos.utils import apply_bitstring, qarray, int_to_bits


class Endianness(IntEnum):
    """Endianness of statevector / unitary output in test helpers."""

    BIG = 0
    LITTLE = 1


type RunnableGuppy = GuppyFunctionDefinition | Package | EmulatorInstance


def _get_emulator(main: RunnableGuppy, n_total_qubits: int) -> EmulatorInstance:
    """Take a guppy program and convert it to an EmulatorInstance.

    Takes either a guppy entrypoint, or a precompiled program and returns
    an EmulatorInstance to be run, compiling only if necessary. This allows
    helpers to take programs either pre or post compilation.
    """
    match main:
        case GuppyFunctionDefinition():
            emulator = main.emulator(n_total_qubits)  # compile
        case Package():
            emulator = EmulatorInstance(_instance=build(main), _n_qubits=n_total_qubits)
        case EmulatorInstance():
            emulator = main
    return emulator


def get_statevector(
    main: RunnableGuppy,
    n_qubits: int,
    *,
    main_args_dict: dict[str, Any] | None = None,
) -> NDArray[np.complex128]:
    """Get the state vector for guppy main program.

    The main program must name the state result as "result_state".

    Args:
        main: The guppy main function to execute
        n_qubits: The number of qubits in the circuit.
        main_args_dict: arguments to pass to guppy main function.

    Returns:
        NDArray[np.complex128]: The state vector representing the circuit.

    """
    emulator = _get_emulator(main, n_qubits)

    if main_args_dict is not None:
        shots = emulator.run(**main_args_dict)
    else:
        shots = emulator.run()
    for shot in shots.results:
        states = Quest.extract_states_dict(shot.entries)
    return states["result_state"].state


def _emulator_with_bit_input[n_state: nat](
    circ: GuppyFunctionDefinition[[array[qubit, n_state]], None],
    n_qubits: int,
    n_extra_qubits: int = 0,
) -> EmulatorInstance:
    @guppy
    @no_type_check
    def main(bits: array[bool, n_qubits]) -> None:
        qreg = qarray(n_qubits)
        apply_bitstring(qreg, bits)
        circ(qreg)
        state_output("result_state", qreg)
        discard_array(qreg)

    return main.emulator(n_qubits + n_extra_qubits)


def _compute_sv_bits_phaseless[n_state: nat](
    args: tuple[
        int,
        GuppyFunctionDefinition[[array[qubit, n_state]], None],
        int,
        int,
    ],
) -> NDArray[np.complex128]:
    i, circ, n_qubits, n_extra_qubits = args
    binary_index = [(i >> bit) & 1 == 1 for bit in range(n_qubits)]

    @guppy
    @no_type_check
    def main() -> None:
        idx = binary_index
        qreg = qarray(n_qubits)
        apply_bitstring(qreg, idx)
        circ(qreg)
        state_output("result_state", qreg)
        discard_array(qreg)

    HUGR = main.compile()
    runner = build(HUGR)

    shots = QsysResult(
        runner.run_shots(
            simulator=Quest(),
            n_qubits=n_qubits + n_extra_qubits,
            n_shots=1,
        )
    )
    for shot in shots.results:
        states = Quest.extract_states_dict(shot.entries)
    _statevector_with_zeroed_ancilla(states["result_state"])
    return switch_endianness(states["result_state"].get_single_state())


def get_unitary[n_state: nat](
    circ: GuppyFunctionDefinition[[array[qubit, n_state]], None],
    n_qubits: int,
    endianness: Endianness = Endianness.BIG,
    n_extra_qubits: int = 0,
) -> NDArray[np.complex128]:
    """Compute the unitary by iterating shots in a loop.

    This function is a wrapper around the get_statevector function, which
    computes the statevector for each basis state. The resulting statevectors
    are then combined to form the unitary matrix.

    Args:
        circ: The circuit
            function to be executed.
        n_qubits: The number of qubits in the circuit.
        endianness: Convert to big endian (default)
        n_extra_qubits: Extra qubits to add to the emulator pool for internal
            ancilla. These ancilla must be returned to zero.

    Returns:
        NDArray[np.complex128]: The unitary matrix representing the circuit.

    """
    size = 2**n_qubits
    emulator = _emulator_with_bit_input(circ, n_qubits, n_extra_qubits)
    flat = np.fromiter(
        (
            elem
            for i in range(size)
            for elem in _statevector_with_zeroed_ancilla(
                Quest.extract_states_dict(
                    emulator.run(bits=int_to_bits(i, n_qubits)).results[0].entries
                )["result_state"]
            )
        ),
        dtype=np.complex128,
        count=size * size,
    )
    mat = flat.reshape(size, size).T
    # fix the phases from the x's preparing the initial bitstring
    for i in range(size):
        mat[:, i] *= np.exp(1j * np.pi / 2 * hamming_weight(i, n_qubits))

    if endianness == Endianness.BIG:
        return switch_matrix_endianness(mat)
    else:
        return mat


def get_unitary_assumed_phase[n_state: nat](
    circ: GuppyFunctionDefinition[[array[qubit, n_state]], None],
    n_qubits: int,
    endianness: Endianness = Endianness.BIG,
    n_extra_qubits: int = 0,
) -> NDArray[np.complex128]:
    """Compute the unitary by iterating shots in a loop.

    Similar to get_unitary, but uses _compute_sv_bits_phaseless which
    directly accesses the Quest result with .get_single_state instead
    of get_statevector. This is needed to apply mem_swaps, but loses
    phase information.


    Args:
        circ: The circuit
            function to be executed.
        n_qubits: The number of qubits in the circuit.
        endianness: Convert to big endian (default)
        n_extra_qubits: Extra qubits to add to the simulator pool for internal
            ancilla. These ancilla must be returned to zero.

    Returns:
        NDArray[np.complex128]: The unitary matrix representing the circuit.

    """
    size = 2**n_qubits
    flat = np.fromiter(
        (
            elem
            for i in range(size)
            for elem in _compute_sv_bits_phaseless((i, circ, n_qubits, n_extra_qubits))
        ),
        dtype=np.complex128,
        count=size * size,
    )
    mat = flat.reshape(size, size).T
    # note contrast to get_unitary phase shifts

    if endianness == Endianness.BIG:
        return switch_matrix_endianness(mat)
    else:
        return mat


def _build_projected_basis_main(
    circ: GuppyFunctionDefinition,
    n_state_qubits: int,
    reg_names: list[str],
    pre_select_dict: dict[str, list[bool]],
) -> tuple[GuppyFunctionDefinition[[], None], int]:
    # Tag qreg separately so internal ancilla are excluded from matrix columns.
    project_register_lens = [len(pre_select_dict[name]) for name in reg_names]

    if len(project_register_lens) == 1:
        pre_selection_0 = pre_select_dict[reg_names[0]]

        @guppy
        @no_type_check
        def main(bits: array[bool, n_state_qubits]) -> None:
            pre_0 = pre_selection_0
            qreg = qarray(n_state_qubits)
            proj_qreg_0 = qarray(comptime(len(pre_selection_0)))

            apply_bitstring(proj_qreg_0, pre_0)
            apply_bitstring(qreg, bits)

            circ(proj_qreg_0, qreg)
            barrier(proj_qreg_0, qreg)
            state_output("projection_0", proj_qreg_0)
            state_output("projected_state", qreg)
            discard_array(qreg)
            discard_array(proj_qreg_0)

    elif len(project_register_lens) == 2:
        pre_selection_0 = pre_select_dict[reg_names[0]]
        pre_selection_1 = pre_select_dict[reg_names[1]]

        @guppy
        @no_type_check
        def main(bits: array[bool, n_state_qubits]) -> None:
            pre_0 = pre_selection_0
            pre_1 = pre_selection_1
            qreg = qarray(n_state_qubits)
            proj_qreg_0 = qarray(comptime(len(pre_selection_0)))
            proj_qreg_1 = qarray(comptime(len(pre_selection_1)))

            apply_bitstring(proj_qreg_0, pre_0)
            apply_bitstring(proj_qreg_1, pre_1)
            apply_bitstring(qreg, bits)

            circ(proj_qreg_0, proj_qreg_1, qreg)

            barrier(proj_qreg_0, proj_qreg_1, qreg)
            state_output("projection_0", proj_qreg_0)
            state_output("projection_1", proj_qreg_1)
            state_output("projected_state", qreg)
            discard_array(qreg)
            discard_array(proj_qreg_0)
            discard_array(proj_qreg_1)

    elif len(project_register_lens) == 3:
        pre_selection_0 = pre_select_dict[reg_names[0]]
        pre_selection_1 = pre_select_dict[reg_names[1]]
        pre_selection_2 = pre_select_dict[reg_names[2]]

        @guppy
        @no_type_check
        def main(bits: array[bool, n_state_qubits]) -> None:
            pre_0 = pre_selection_0
            pre_1 = pre_selection_1
            pre_2 = pre_selection_2
            qreg = qarray(n_state_qubits)
            proj_qreg_0 = qarray(comptime(len(pre_selection_0)))
            proj_qreg_1 = qarray(comptime(len(pre_selection_1)))
            proj_qreg_2 = qarray(comptime(len(pre_selection_2)))

            apply_bitstring(proj_qreg_0, pre_0)
            apply_bitstring(proj_qreg_1, pre_1)
            apply_bitstring(proj_qreg_2, pre_2)
            apply_bitstring(qreg, bits)

            circ(proj_qreg_0, proj_qreg_1, proj_qreg_2, qreg)

            barrier(proj_qreg_0, proj_qreg_1, proj_qreg_2, qreg)
            state_output("projection_0", proj_qreg_0)
            state_output("projection_1", proj_qreg_1)
            state_output("projection_2", proj_qreg_2)
            state_output("projected_state", qreg)
            discard_array(qreg)
            discard_array(proj_qreg_0)
            discard_array(proj_qreg_1)
            discard_array(proj_qreg_2)

    else:
        raise ValueError(
            "get_unitary_projected supports at most 3 projected registers."
        )

    return main, n_state_qubits + sum(project_register_lens)


def _emulator_with_bit_input_projected(
    circ: GuppyFunctionDefinition,
    n_state_qubits: int,
    post_select_dict: dict[str, list[bool]],
    pre_select_dict: dict[str, list[bool]],
    n_extra_qubits: int = 0,
) -> tuple[EmulatorInstance, int]:
    reg_names = list(post_select_dict.keys())

    main, n_total_qubits = _build_projected_basis_main(
        circ, n_state_qubits, reg_names, pre_select_dict
    )

    return main.emulator(
        n_total_qubits + n_extra_qubits
    ), n_total_qubits + n_extra_qubits


def statevector_projected_selene(
    main: RunnableGuppy,
    n_qubits: int,
    post_select_dict: dict[str, list[bool]],
    returned_specified_qubits: list[int] | None = None,
    renormalize: bool = True,
    *,
    returned_state_tag: str | None = None,
    main_args_dict: dict[str, Any] | None = None,
) -> SeleneQuestState:
    """Project named ``state_result`` registers in sequence.

    This executes ``main``, extracts the named ``state_result`` entries,
    and then applies ``project_state_onto_bitstring`` repeatedly in the order the
    register names appear in ``post_select_dict``. Each projection can pass the
    remaining specified qubits from the next register so that later projections
    operate on the expected subspace.

    All state results should be disjoint in their specified qubits. In practice,
    it is safest to use them at the same point in the program over all
    qubits, to avoid accidentally overlapping qubits.

    Args:
        main: The guppy main function to execute.
        n_qubits: The total number of qubits in the compiled program.
        post_select_dict: Mapping from ``state_result`` tag to the boolean
            bitstring to project that register onto. The insertion order of this
            dictionary determines the projection order.
        returned_specified_qubits: Optional specified-qubit list to assign to the
            final returned state after all projections have been applied.
        returned_state_tag: Optional state-result tag whose qubits should be marked
            as the specified qubits in the returned state.
        renormalize: Whether to renormalize the state after each projection step.
        main_args_dict: arguments to pass to guppy main function.

    Returns:
        SeleneQuestState: The final state after all requested register
        projections have been applied.

    """
    emulator = _get_emulator(main, n_qubits)
    if main_args_dict is not None:
        res = emulator.run(**main_args_dict)
    else:
        res = emulator.run()
    # Extract all named state_result registers from a single emulator shot.
    states = Quest.extract_states_dict(res.results[0].entries)

    if returned_state_tag is not None:
        if returned_specified_qubits is not None:
            raise ValueError(
                "Specify either returned_specified_qubits or returned_state_tag, "
                "not both."
            )
        if returned_state_tag not in states:
            raise ValueError(
                f"Returned state tag {returned_state_tag!r} is not available."
            )
        returned_specified_qubits = states[returned_state_tag].specified_qubits

    if not set(post_select_dict.keys()).issubset(set(states.keys())):
        raise ValueError(
            "Post selection keys must be a subset of the available state keys"
        )
    regs = list(post_select_dict.keys())
    ps_bools = list(post_select_dict.values())

    # Collect the specified-qubit indices for each register in projection order.
    specified_qubits_list = [states[reg].specified_qubits for reg in regs]

    for ps, qubits in zip(ps_bools, specified_qubits_list, strict=True):
        if len(ps) != len(qubits):
            raise ValueError(
                "Length of post selection bools does not match the number of "
                "specified qubits in the next projection register."
            )

    # After projecting one register away, the next register's specified qubits
    # become the reference frame for the following projection step. The final
    # step can optionally replace them with returned_specified_qubits.
    next_projection_qubits = [
        *specified_qubits_list[1:],
        returned_specified_qubits,
    ]

    state = states[regs[0]]

    # Apply the requested projections sequentially, carrying forward the reduced
    # state after each step.
    for ps, new_specified_qubits in zip(ps_bools, next_projection_qubits, strict=True):
        state = project_state_onto_bitstring(
            state,
            ps,
            new_specified_qubits=new_specified_qubits,
            renormalize=renormalize,
        )
        state = state.state
    return state


def _statevector_with_zeroed_ancilla(state: SeleneQuestState) -> NDArray[np.complex128]:
    """Extract specified qubits while requiring all other qubits to be zero."""
    n_specified = len(state.specified_qubits)
    n_unspecified = state.total_qubits - n_specified
    local_specified_qubits = (
        state._reindex_global_to_local(state.specified_qubits)
        if isinstance(state, SubQuestState)
        else state.specified_qubits
    )

    permutation_lhs = []
    permutation_rhs = [-1] * n_specified
    for qubit_id, bit_index in enumerate(reversed(range(state.total_qubits))):
        if qubit_id in local_specified_qubits:
            specified_index = local_specified_qubits.index(qubit_id)
            permutation_rhs[specified_index] = bit_index
        else:
            permutation_lhs.append(bit_index)

    # Group unspecified ancilla by row and state amplitudes by column.
    state_by_ancilla = np.transpose(
        state.state.reshape([2] * state.total_qubits),
        permutation_lhs + permutation_rhs,
    ).reshape((2**n_unspecified, 2**n_specified))
    if not np.allclose(state_by_ancilla[1:], 0, atol=1e-10):
        raise ValueError("Internal ancilla qubits were not returned to the zero state.")
    # Keep the zero-ancilla branch without renormalizing or changing its phase.
    return switch_endianness(state_by_ancilla[0])


def get_statevector_projected(
    main: RunnableGuppy,
    n_qubits: int,
    post_select_dict: dict[str, list[bool]],
    renormalize: bool = True,
    *,
    main_args_dict: dict[str, Any] | None = None,
) -> NDArray[np.complex128]:
    """Return a statevector after sequential register post-selection.

    This is a user friendly convenience wrapper around
    ``statevector_projected_selene`` that exposes only the final NumPy
    statevector instead of the full ``SeleneQuestState`` container. It executes
    the same sequence of projections on the named registers and returns the
    final projected statevector. Unlike
    ``statevector_projected_selene``, this wrapper does not expose the
    option to return final specified qubits, so the result is always the
    statevector over the remaining unprojected qubits. Use
    ``statevector_projected_selene`` directly when you need that extra
    control.

    Args:
        main: The guppy main function to execute.
        n_qubits: The total number of qubits in the compiled program.
        post_select_dict: Mapping from ``state_output`` tag to the boolean
            bitstring to project that register onto, applied in insertion order.
        renormalize: Whether to renormalize the state after each projection step.
        main_args_dict: arguments to pass to guppy main function.

    Returns:
        NDArray[np.complex128]: The projected statevector.

    """
    return statevector_projected_selene(
        main,
        n_qubits,
        post_select_dict,
        renormalize=renormalize,
        main_args_dict=main_args_dict,
    ).state


def get_unitary_projected(
    circ: GuppyFunctionDefinition,
    n_state_qubits: int,
    post_select_dict: dict[str, list[bool]],
    pre_select_dict: dict[str, list[bool]] | None = None,
    endianness: Endianness = Endianness.BIG,
    n_extra_qubits: int = 0,
) -> NDArray[np.complex128]:
    """Project into one or more non-state registers of a guppy circuit.

    This function will project into a sub-block of a unitary matrix
    defined by the matrix element mapping the pre-selected input to the
    post-selected output. Each projected register starts in the zero state
    when no ``pre_select_dict`` is provided.

    The circuit must take one, two, or three projected registers followed by the
    state register. The user-supplied circuit is responsible for applying any
    ``state_result`` tags to those projected registers and discarding the
    allocated registers after calling ``circ``.

    Args:
        circ: The circuit function to be executed. It must accept the projected
            registers and a state register, and it must emit the
            ``state_result`` tags referenced by ``post_select_dict``.
        n_state_qubits: The number of qubits in the state block.
        post_select_dict: Mapping from projected register name to the bitstring
            to project onto. The insertion order determines projected register
            order when calling ``circ``.
        pre_select_dict: Optional mapping from projected register name to the
            bitstring used to prepare each projected register before applying
            ``circ``. If None, all projected registers are initialized to zero.
        endianness: Convert to big endian (default)
        n_extra_qubits: Extra qubits to add to the emulator pool beyond the
            explicitly named registers. Use this when the circuit internally
            allocates ancilla qubits via ``qubit()``.

    Returns:
        NDArray[np.complex128]: The projected matrix.

    """
    reg_names = list(post_select_dict.keys())
    if pre_select_dict is None:
        pre_select_dict = {
            reg_name: [False] * len(post_select_dict[reg_name])
            for reg_name in reg_names
        }
    elif set(pre_select_dict.keys()) != set(post_select_dict.keys()):
        raise ValueError(
            "Pre-selection keys must exactly match the post-selection keys."
        )

    for reg_name in reg_names:
        if len(pre_select_dict[reg_name]) != len(post_select_dict[reg_name]):
            raise ValueError(
                f"Pre- and post-selection bitstrings for register {reg_name} "
                "must have the same length."
            )

    emulator, n_total_qubits = _emulator_with_bit_input_projected(
        circ, n_state_qubits, post_select_dict, pre_select_dict, n_extra_qubits
    )

    internal_post_select_dict = {
        f"projection_{i}": list(post_select_dict[reg_name])
        for i, reg_name in enumerate(reg_names)
    }

    size = 2**n_state_qubits
    flat = np.fromiter(
        (
            elem
            for i in range(size)
            for elem in _statevector_with_zeroed_ancilla(
                statevector_projected_selene(
                    emulator,
                    n_total_qubits,
                    internal_post_select_dict,
                    returned_state_tag="projected_state",
                    main_args_dict={"bits": int_to_bits(i, n_state_qubits)},
                    renormalize=False,
                )
            )
        ),
        dtype=np.complex128,
        count=size * size,
    )

    mat = flat.reshape(size, size)
    mat = mat.T
    # fix the phases from the x's preparing the initial bitstring
    for i in range(size):
        mat[:, i] *= np.exp(1j * np.pi / 2 * hamming_weight(i, n_state_qubits))
    if endianness == Endianness.BIG:
        return switch_matrix_endianness(mat)
    else:
        return mat


def switch_endianness(vec: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Switch the endianness of a vector.

    Args:
        vec: The input 2^n vec.

    Returns:
        NDArray[np.complex128]: The vec with reversed bit order.

    """
    n = int(np.log2(vec.size))
    if 1 << n != vec.size:
        raise ValueError("Vector length must be a power of two.")
    indices = np.arange(vec.size, dtype=np.uint64)
    reversed_indices = np.zeros_like(indices)
    for _ in range(n):
        reversed_indices = (reversed_indices << 1) | (indices & 1)
        indices >>= 1
    return np.take(vec, reversed_indices)


def _bit_reverse_perm(size: int) -> NDArray[np.uint64]:
    n = int(np.log2(size))
    if 1 << n != size:
        raise ValueError("size must be a power of two")
    idx = np.arange(size, dtype=np.uint64)
    rev = np.zeros_like(idx)
    tmp = idx.copy()
    for _ in range(n):
        rev = (rev << 1) | (tmp & 1)
        tmp >>= 1
    return rev.astype(np.uint64)


def switch_matrix_endianness(mat: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Switch the endianness of a matrix.

    Args:
        mat: The input 2^n x 2^n matrix.

    Returns:
        NDArray[np.complex128]: The mat with reversed bit order in both dimensions.

    """
    size = mat.shape[0]
    if mat.shape != (size, size):
        raise ValueError("Matrix must be square")
    perm = _bit_reverse_perm(size)

    # exact sequential 1-D passes (row, then column), but vectorized
    tmp = np.empty_like(mat)
    out = np.empty_like(mat)
    np.take(mat, perm, axis=0, out=tmp)
    np.take(tmp, perm, axis=1, out=out)
    return out

    # alternative that is slower but uses less memory
    # size = mat.shape[0]
    # for i in range(size):
    #     mat[:, i] = switch_endianness(mat[:, i])
    # for i in range(size):
    #     mat[i, :] = switch_endianness(mat[i, :])

    # return mat


def align_phase(
    vec: NDArray[np.complex128], threshold: float = 1e-8
) -> NDArray[np.complex128]:
    """Align the global phase of a vector so that the first large element is real."""
    idx = np.flatnonzero(np.abs(vec) > threshold)
    i = idx[0]
    phase: np.complex128 = np.exp(-1j * np.angle(vec.flat[i]))
    return vec * phase


def assert_allclose_ignorephase(
    a: NDArray[np.complex128], b: NDArray[np.complex128], threshold: float = 1e-8
) -> None:
    """Stopgap measure until simulations respect global phase."""
    if np.allclose(np.linalg.norm(a), 0):
        np.testing.assert_allclose(a, b)

    a_aligned = align_phase(a)
    b_aligned = align_phase(b)

    np.testing.assert_allclose(a_aligned, b_aligned, atol=threshold)


def assert_cntrl_unitary(
    circ: GuppyFunctionDefinition,
    expected_active_unitary: NDArray[np.complex128],
    n_state_qubits: int,
    post_select_dict: dict[str, list[bool]],
    *,
    control_name: str = "control",
    endianness: Endianness = Endianness.BIG,
    n_extra_qubits: int = 0,
    threshold: float = 1e-8,
) -> None:
    r"""Check the coherent blocks of a controlled unitary.

    The circuit must conjugate its control with Hadamard gates and accept that
    one-qubit control register before any registers in ``post_select_dict``.
    If its active operation is :math:`A`, projecting the control onto zero and
    one must respectively produce :math:`(I + A) / 2` and
    :math:`(I - A) / 2`.

    """
    if control_name in post_select_dict:
        raise ValueError(
            f"post_select_dict must not contain control register {control_name!r}"
        )

    actual_blocks = np.stack(
        [
            get_unitary_projected(
                circ,
                n_state_qubits,
                {control_name: [control], **post_select_dict},
                endianness=endianness,
                n_extra_qubits=n_extra_qubits,
            )
            for control in (False, True)
        ]
    )
    identity = np.eye(2**n_state_qubits)
    expected_blocks = np.stack(
        [
            (identity + expected_active_unitary) / 2,
            (identity - expected_active_unitary) / 2,
        ]
    )
    assert_allclose_ignorephase(actual_blocks, expected_blocks, threshold)


def project_state_onto_bitstring(
    state: SeleneQuestState | SubQuestState,
    bitstring: list[bool],
    zero_threshold: float = 1e-12,
    new_specified_qubits: list[int] | None = None,
    renormalize: bool = True,
) -> TracedState[SubQuestState]:
    """Projects a state onto the desired bitstring on the specified qubits.

    Args:
        state: Output quest state
        bitstring: Desired bitstring on state.specified_qubits with bitstring[0] the MSB
        zero_threshold: point below which the output state counts as having 0 amplitude
        new_specified_qubits: List of indices of qubits you are interested in
            after the projection, these must be disjoint with the qubits projected on
        renormalize: Whether to renormalize the projected state

    Returns:
        TracedState, which contains a SubQuestState, a statevector over the
        qubits which have not been projected on, along with the probability of
        the postselection succeeding. The output SubQuestState has
        specified qubits determined by new_specified_qubits, with
        specified_qubits=[] if new_specified_qubits=None, and keeps track
        of which qubits have been projected out so far.

    """
    if len(bitstring) != len(state.specified_qubits):
        raise ValueError(f"""Projection bitstring {bitstring} must have the same length
        as specified qubits {state.specified_qubits}""")

    if isinstance(state, SubQuestState):
        if not set(state._projected_out_qubits).isdisjoint(state.specified_qubits):
            raise ValueError(
                "Trying to project out qubits that have already been discarded"
            )

    state_tensor = state.state.reshape([2] * state.total_qubits)
    # move all specified qubits to the end, in the user-specified order
    n_specified = len(state.specified_qubits)
    n_unspecified = state.total_qubits - n_specified
    permutation_lhs = []
    permutation_rhs = [-1 for _ in range(n_specified)]
    # get the specified qubits in the local frame if a SubQuestState
    local_specified_qubits = (
        state._reindex_global_to_local(state.specified_qubits)
        if isinstance(state, SubQuestState)
        else state.specified_qubits
    )
    # Note: QuEST uses the convention that qubit 0 is the least significant bit.
    # Thus to iterate over qubits and corresponding statevector indices, we need
    # to iterate from left to right in one, right to left in the other.
    for qubit_id, bit_index in enumerate(reversed(range(state.total_qubits))):
        if qubit_id in local_specified_qubits:
            specified_index = local_specified_qubits.index(qubit_id)
            permutation_rhs[specified_index] = bit_index
        else:
            permutation_lhs.append(bit_index)
    if -1 in permutation_rhs:
        raise ValueError("All specified qubits must be assigned")
    permutation = permutation_lhs + permutation_rhs
    permuted = np.transpose(state_tensor, permutation)
    # state_tensor is now in the shape ([2]*n_unspecified + [2]*n_specified).
    # reshape to a matrix
    reshaped = permuted.reshape((2**n_unspecified, 2**n_specified))

    # get the projected state
    index = sum(2**i if bit else 0 for i, bit in enumerate(reversed(bitstring)))
    projected_state = switch_endianness(reshaped[:, index])

    # renormalize
    if renormalize:
        projected_norm = np.linalg.norm(projected_state)
    else:
        projected_norm = 1.0

    # work out indices of new specified qubits now that we have projected out old ones
    if new_specified_qubits is None:
        new_specified_qubits = []

    if projected_norm > zero_threshold:
        projected_state = projected_state / projected_norm

    # build output SubQuestState and TracedState
    if isinstance(state, SubQuestState):
        projected_out_qubits = state.specified_qubits + state._projected_out_qubits
    else:
        projected_out_qubits = state.specified_qubits

    sqs = SubQuestState(
        projected_state,
        total_qubits=n_unspecified,
        specified_qubits=new_specified_qubits,
        projected_out_qubits=projected_out_qubits,
    )

    return TracedState(float(projected_norm**2), sqs)


def hamming_weight(i: int, n: int) -> int:
    """Compute the hamming weight of i as an n bit binary string."""
    return sum((i >> bit) & 1 == 1 for bit in range(n))


def get_total_state_on_only_specified_registers(
    states: dict[str, SeleneQuestState], result_tags: list[str]
) -> tuple[SubQuestState, dict[str, list[int]]]:
    """Extract the total state over all registers specified by result_tags.

    This function makes several assumptions, firstly the state_results in your guppy
    are located in the same place, and have tags matching result_tags and additionally
    that the total state of these registers is pure (unentangled with any other qubits).

    Args:
        states (dict[str, SeleneQuestState]): state dictionary, as returned
        by Quest.extract_states_dict.
        result_tags (list[str]): List of state_result tags corresponding to the desired
        registers.

    Returns:
        A SubQuestState for the pure state over all specified registers, with
        specified_qubits equal to all those qubits.
        A dictionary mapping result tags to specified_qubit lists.

    """
    specified_qubits_dict = {}
    for result_tag in result_tags:
        specified_qubits_dict[result_tag] = states[result_tag].specified_qubits

    total_specified_qubits = list(chain.from_iterable(specified_qubits_dict.values()))
    total_specified_qubits.sort()

    total_state = states[result_tags[0]]
    total_state.specified_qubits = total_specified_qubits
    # get the state without ancilla qubits (rightly requires that they are unentangled)
    try:
        total_state = total_state.get_single_state()
    except ValueError as err:
        raise ValueError(
            f"Getting state only on registers {result_tags} failed "
            + "as they are entangled with other qubits."
        ) from err
    # fix the specified qubits we just changed
    states[result_tags[0]].specified_qubits = specified_qubits_dict[result_tags[0]]

    # get_single_state has switched the endianness and since we want to feed it
    # back into a SubQuestState we will need to switch it back
    projected_out_qubits = [
        i
        for i in range(states[result_tags[0]].total_qubits)
        if i not in total_specified_qubits
    ]
    total_state = SubQuestState(
        state=switch_endianness(total_state),
        total_qubits=len(total_specified_qubits),
        specified_qubits=total_specified_qubits,
        projected_out_qubits=projected_out_qubits,
    )

    return total_state, specified_qubits_dict


class SubQuestState:
    """Wrapper for SeleneQuestState that represents a pure substate of a system.

    Has the same public interface as SeleneQuestState.
    Pure substates appear when doing projections or discarding unentangled qubits,
    but doing sequences of these operations requires carefully keeping track of qubit
    indices within the sim on a global and local level.
    This wrapper handles the reindexing by silently keeping track of qubits that have
    already been "projected out".
    """

    _wrapped: SeleneQuestState
    total_qubits: int
    _projected_out_qubits: list[int]

    def __init__(
        self,
        state: NDArray[np.complex128],
        total_qubits: int,
        specified_qubits: list[int],
        projected_out_qubits: list[int],
    ):
        """Produce a substate by providing a little endian statevector and indices.

        Take in a statevector array and packs it into a SubQuestState.
        Reindex based on qubits indices that are no longer present.
        """
        self._wrapped = SeleneQuestState(state, total_qubits, specified_qubits)
        self._projected_out_qubits = projected_out_qubits
        self.total_qubits = total_qubits
        self.specified_qubits = specified_qubits  # reindexing in setter

    @property
    def state(self) -> np.ndarray:
        """Gets full statevector."""
        return self._wrapped.state

    @property
    def specified_qubits(self) -> list[int]:
        """Gets specified qubits in global frame."""
        return self._reindex_local_to_global(self._wrapped.specified_qubits)

    @specified_qubits.setter
    def specified_qubits(self, spec_qubits_outer: list[int]) -> None:
        """Set specified qubits of inner SeleneQuestState.

        Reindexes using the fact that some qubits have already been removed.
        """
        if not set(spec_qubits_outer).isdisjoint(self._projected_out_qubits):
            raise ValueError(
                "Setting specified qubits of SubQuestState failed "
                "as global indices overlap with already projected out qubits."
            )

        # Reindex spec_qubits_outer to account for projected out qubits
        spec_qubits_reindexed = self._reindex_global_to_local(spec_qubits_outer)

        if any(q >= self.total_qubits or q < 0 for q in spec_qubits_reindexed):
            raise ValueError(
                f"All new_specified_qubits must be in the range "
                f"[0, {self.total_qubits - 1}], "
                f"but got {spec_qubits_reindexed}"
            )

        self._wrapped.specified_qubits = spec_qubits_reindexed

    def _reindex_global_to_local(self, global_idxs: list[int]):
        local_idxs = []
        for qubitidx in global_idxs:
            decrement = sum(1 for pq in self._projected_out_qubits if pq < qubitidx)
            local_idxs.append(qubitidx - decrement)
        return local_idxs

    def _reindex_local_to_global(self, local_idxs: list[int]):
        global_idxs = []
        for local_idx in local_idxs:
            global_idx = local_idx
            for pq in sorted(self._projected_out_qubits):
                if pq <= global_idx:
                    global_idx += 1
            global_idxs.append(global_idx)
        return global_idxs

    def get_density_matrix(self, zero_threshold: float = 1e-12) -> np.ndarray:
        """Get the density matrix of the subsystem given by the specified qubits."""
        return self._wrapped.get_density_matrix(zero_threshold=zero_threshold)

    def get_state_vector_distribution(
        self, zero_threshold=1e-12
    ) -> list[TracedState[np.ndarray]]:
        """Get the statevector distribution of the subsystem."""
        return self._wrapped.get_state_vector_distribution(
            zero_threshold=zero_threshold
        )

    def get_single_state(self, zero_threshold=1e-12) -> np.ndarray:
        """Get the pure state of the subsystem if possible."""
        return self._wrapped.get_single_state(zero_threshold=zero_threshold)

    def get_dirac_notation(self, zero_threshold=1e-12) -> list[TracedState]:
        """Get dirac notation for the state of the subsystem."""
        return self._wrapped.get_dirac_notation(zero_threshold=zero_threshold)

    def get_single_dirac_notation(self, zero_threshold=1e-12) -> TracedState:
        """Get dirac notation of the pure subsystem if possible."""
        return self._wrapped.get_single_dirac_notation(zero_threshold=zero_threshold)


def extract_state_branches_in_superposition(
    states: dict[str, SeleneQuestState],
    branch_tag: str,
    result_tags: list[str],
    branch_bitstrings: list[list[bool]],
) -> dict[tuple[bool, ...], TracedState[SeleneQuestState]]:
    """Project a superposition state onto every requested branch of a single register.

    Args:
        states: State dictionary as returned by ``Quest.extract_states_dict``.
        branch_tag: State-result tag of the register used to select branches.
        result_tags: State-result tags to keep after each branch projection.
        branch_bitstrings: Bitstrings to project onto for the branch register.

    Returns:
        A dictionary mapping each branch bitstring to its projected state.
        Each projected state lives on the remaining ``result_tags`` registers only,
        with the remaining registers encoded directly in
        ``projected_state.state.state``.

    """
    total_state, specified_qubits_dict = get_total_state_on_only_specified_registers(
        states, [branch_tag, *result_tags]
    )
    projected_states = {}
    for bitstring in branch_bitstrings:
        # Project onto one branch of the branch register; because ``total_state`` only
        # contains ``branch_tag`` and ``result_tags``, the remaining state is exactly
        # the requested result registers.
        total_state.specified_qubits = specified_qubits_dict[branch_tag]
        projected_states[tuple(bitstring)] = project_state_onto_bitstring(
            total_state, bitstring
        )

    return projected_states


def chebyshev_power_matrix(
    mat: NDArray[np.complex128], power: int
) -> NDArray[np.complex128]:
    """Return the Chebyshev polynomial of a Hermitian matrix."""
    coeffs = [0] * power + [1]
    eigenvalues, eigenvectors = np.linalg.eigh(mat)
    chebyshev_eigenvalues = chebval(eigenvalues, coeffs)
    return eigenvectors @ np.diag(chebyshev_eigenvalues) @ eigenvectors.conj().T
