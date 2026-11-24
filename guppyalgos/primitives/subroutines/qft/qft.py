"""Quantum Fourier Transform."""

from typing import no_type_check

from guppylang import guppy
from guppylang.std.angles import pi
from guppylang.std.builtins import array, control, nat
from guppylang.std.quantum import discard, h, qubit

from guppyalgos.primitives.gate_decompositions.cnx.cnx import CNXPartialAndLadder
from guppyalgos.primitives.utilities.gates import apply_phase, swap


@guppy(controllable=True)
@no_type_check
def _qft_kernel[n_qft: nat](qs: array[qubit, n_qft]) -> None:
    """Apply the forward QFT ladder."""
    for i in range(n_qft - 1, -1, -1):
        h(qs[i])
        for j in range(i):
            with control(qs[j]):
                apply_phase(qs[i], pi / 2 ** (i - j))
    for i in range(n_qft // 2):
        swap(qs[i], qs[n_qft - i - 1])


@guppy(controllable=True)
@no_type_check
def _iqft_kernel[n_qft: nat](qs: array[qubit, n_qft]) -> None:
    """Apply the inverse QFT ladder with explicitly reversed loops."""
    for i in range(n_qft // 2):
        swap(qs[i], qs[n_qft - i - 1])
    for i in range(n_qft):
        for j in range(i):
            with control(qs[j]):
                apply_phase(qs[i], -pi / 2 ** (i - j))
        h(qs[i])


@guppy.unitary
class qft[n_qft: nat]:  # noqa: D101
    @guppy
    @no_type_check
    def __call__(qs: array[qubit, n_qft]) -> None:
        """In-place quantum Fourier transform on qs.

        Args:
            qs: Qubit register to perform the QFT on.

        Returns:
            None

        """
        _qft_kernel(qs)

    @guppy
    @no_type_check
    def controlled[n_controls: nat](
        qs: array[qubit, n_qft], controls: array[qubit, n_controls]
    ) -> None:
        """Apply QFT under external controls.

        For two or more external controls, this combines them into a temporary
        conjunction flag. For more than two controls, the CNX ladder also
        allocates a workspace register. All temporary qubits are uncomputed
        and discarded before this method returns.
        """
        if n_controls == 1:
            with control(controls[0]):
                _qft_kernel(qs)
        else:
            temp_control = qubit()
            cnx_ancilla = CNXPartialAndLadder.compute(controls, temp_control)
            with control(temp_control):
                _qft_kernel(qs)
            CNXPartialAndLadder.uncompute(controls, cnx_ancilla, temp_control)
            discard(temp_control)

    @guppy
    @no_type_check
    def daggered(qs: array[qubit, n_qft]) -> None:  # noqa: D102
        # Custom dagger implementation required only until Guppy can reverse loops.
        _iqft_kernel(qs)

    @guppy
    @no_type_check
    def ctrl_daggered[n_controls: nat](
        qs: array[qubit, n_qft], controls: array[qubit, n_controls]
    ) -> None:
        """Apply inverse QFT under external controls.

        For two or more external controls, this combines them into a temporary
        conjunction flag. For more than two controls, the CNX ladder also
        allocates a workspace register. All temporary qubits are uncomputed
        and discarded before this method returns.
        """
        # Custom dagger implementation required only until Guppy can reverse loops.
        if n_controls == 1:
            with control(controls[0]):
                _iqft_kernel(qs)
        else:
            temp_control = qubit()
            cnx_ancilla = CNXPartialAndLadder.compute(controls, temp_control)
            with control(temp_control):
                _iqft_kernel(qs)
            CNXPartialAndLadder.uncompute(controls, cnx_ancilla, temp_control)
            discard(temp_control)
