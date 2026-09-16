"""Guppy function builders used to configure HHL."""

from __future__ import annotations

from collections.abc import Sequence
from typing import no_type_check

import numpy as np
import zixy.qubit.pauli as zqp
from guppylang import guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.angles import angle
from guppylang.std.builtins import array, nat
from guppylang.std.quantum import cx, qubit, ry
from hugr.model import Apply, DefineFunc, List as HugrList, Literal

from guppyalgos.algorithms.state_preparation.multiplexor_prep import (
    _CommandType,
    _multiplexed_rotation_commands,
)
from guppyalgos.algorithms.time_evolution.trotter import (
    cntrl_ham_sim_trotter,
    cntrl_trotter_first_order,
)


def register_size(function: GuppyFunctionDefinition, parameter_index: int) -> int:
    """Determine the size of a fixed-size array parameter in a Guppy function."""
    package = function.compile_function().to_model()
    function_node = package.modules[0].root.children[0]
    operation = function_node.operation
    if not isinstance(operation, DefineFunc):
        raise TypeError("Expected a compiled Guppy function definition.")
    signature = operation.symbol.signature
    if not isinstance(signature, Apply) or not isinstance(signature.args[0], HugrList):
        raise TypeError("Expected a concrete Guppy function signature.")
    parameter_type = signature.args[0].parts[parameter_index]
    if (
        not isinstance(parameter_type, Apply)
        or "borrow_array" not in parameter_type.symbol
    ):
        raise TypeError(
            f"Expected parameter {parameter_index + 1} to be a fixed-size array."
        )
    array_size = parameter_type.args[0]
    if not isinstance(array_size, Literal) or not isinstance(array_size.value, int):
        raise TypeError("Expected a concrete array size.")
    return array_size.value


def multiplexed_ry[n_controls: nat](
    angles: Sequence[float],
    n_control_qubits: int,
) -> GuppyFunctionDefinition[[array[qubit, n_controls], qubit], None]:
    r"""Construct an ancilla-free uniformly controlled ``Ry`` rotation.

    For ``angles[k]`` in half-turns, the returned function applies

    .. math::

        \sum_k |k\rangle\langle k|_{\mathrm{controls}} \otimes
        R_y(\pi\,\mathrm{angles}[k])_{\mathrm{target}}.

    The controls are little-endian and the implementation uses a Gray-code
    decomposition containing only ``Ry`` and ``CX`` gates.

    Args:
        angles: One half-turn rotation angle for every control-register basis state.
        n_control_qubits: Number of qubits in the control register.

    Returns:
        A function accepting the control register and rotation target.

    Raises:
        ValueError: If the number of angles does not match the control-register size.

    """
    if len(angles) != 2**n_control_qubits:
        raise ValueError(
            "The number of multiplexed rotation angles must be a power of two."
        )

    command_angles = [
        angles[
            sum(
                ((basis_state >> bit_index) & 1) << (n_control_qubits - bit_index - 1)
                for bit_index in range(n_control_qubits)
            )
        ]
        for basis_state in range(2**n_control_qubits)
    ]
    commands = _multiplexed_rotation_commands(command_angles)

    @guppy.comptime
    @no_type_check
    def multiplexed_ry_fn(
        controls: array[qubit, n_control_qubits], target: qubit
    ) -> None:
        for command, value in commands:
            if command == _CommandType.ROTATION:
                ry(target, angle(value))
            else:
                cx(controls[value], target)

    return multiplexed_ry_fn


def eigenvalue_inversion_angles(
    n_qpe: int,
    rotation_scalar: float,
) -> list[float]:
    """Calculate the half-turn rotation angles for eigenvalue inversion.

    Args:
        n_qpe: Number of clock qubits in QPE.
        rotation_scalar: Constant C scaling the inversion factor C / lambda.

    Returns:
        List of 2**n_qpe half-turn angles for Ry rotation on the ancilla qubit.

    """
    dim = 2**n_qpe
    angles = []
    for k in range(dim):
        if k == 0:
            angles.append(0.0)
            continue
        k_signed = k if k < dim // 2 else k - dim
        ratio = rotation_scalar / k_signed
        ratio = max(-1.0, min(1.0, ratio))
        half_turn = (2.0 / np.pi) * np.arcsin(ratio)
        angles.append(float(half_turn))
    return angles


def create_controlled_hamiltonian_simulation[n_input: nat](
    input_matrix: zqp.RealTermSum,
    time_step: float,
    n_trotter_steps: int = 1,
    n_input_qubits: int | None = None,
) -> GuppyFunctionDefinition[[qubit, array[qubit, n_input], int], None]:
    """Build a signed-powered controlled Hamiltonian simulation for HHL.

    Args:
        input_matrix: Hamiltonian represented as a real Pauli term sum.
        time_step: Trotter evolution time step.
        n_trotter_steps: Number of Trotter steps per unit evolution.
        n_input_qubits: Number of qubits in the system register. Defaults to the
            number of qubits represented by ``input_matrix``.

    Returns:
        A controlled simulation accepting a signed integer power. Positive powers
        apply forward evolution and negative powers apply inverse evolution.

    """
    n_qubits = len(input_matrix.qubits) if n_input_qubits is None else n_input_qubits
    controlled_trotter_step = cntrl_trotter_first_order(
        input_matrix, n_state_qubits=n_qubits
    )
    forward_simulation = cntrl_ham_sim_trotter(
        controlled_trotter_step, n_trotter_steps, time_step, n_qubits
    )
    inverse_simulation = cntrl_ham_sim_trotter(
        controlled_trotter_step, n_trotter_steps, -time_step, n_qubits
    )

    @guppy
    @no_type_check
    def controlled_hamiltonian_simulation(
        control: qubit,
        state_register: array[qubit, n_qubits],
        power: int,
    ) -> None:
        if power >= 0:
            for _ in range(power):
                forward_simulation(control, state_register)
        else:
            for _ in range(-power):
                inverse_simulation(control, state_register)

    return controlled_hamiltonian_simulation


def create_eigenvalue_inversion[n_clock: nat](
    n_qpe: int,
    rotation_scalar: float,
) -> GuppyFunctionDefinition[[array[qubit, n_clock], qubit], None]:
    """Build the clock-conditioned eigenvalue-inversion rotation for HHL.

    Args:
        n_qpe: Number of clock qubits in QPE.
        rotation_scalar: Constant C scaling the inversion factor C / lambda.

    Returns:
        A Guppy function applying the conditioned rotation to a zeroed ancilla.

    """
    angles = eigenvalue_inversion_angles(n_qpe, rotation_scalar)
    return multiplexed_ry(angles, n_qpe)
