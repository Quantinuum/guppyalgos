r"""Dynamic quantum Fourier transform followed by measurement.

This module implements the dynamic QFT+measurement (QFT+M) protocol described
in [1].

References:
    [1] Bäumer, E. et al (2024). Quantum Fourier Transform Using Dynamic Circuits. Phys.
    Rev. Lett. 133, 150602.

"""

from typing import no_type_check

from guppylang.decorator import guppy
from guppylang.std.builtins import array, nat
from guppylang.std.quantum import qubit, h, project_z
from guppylang.std.mem import mem_swap
from guppylang.std.angles import pi
from guppyalgos.primitives.utilities.gates import apply_phase


@guppy
@no_type_check
def qft_and_measure[n_qft: nat](
    qs: array[qubit, n_qft],
) -> array[bool, n_qft]:
    r"""Dynamic QFT followed by measurement.

    This implements the dynamic QFT+measurement (QFT+M) protocol described
    in [1].

    Mid-circuit measurements and classical feed-forward replace the
    controlled-phase gates of the standard unitary QFT.
    The protocol requires one adaptive measurement/feed-forward round per qubit,
    resulting in $n$ rounds and $Theta(n^2)$ classically conditioned single-qubit phase
    corrections, rather than $\Theta(n^2)$ controlled-phase gates
    required by the standard unitary QFT.

    This operation is not a unitary QFT and does not produce a reusable transformed
    quantum state. All qubits in ``qs`` are measured, and the measurement results are
    returned in little-endian order. The ``qs`` register should be discarded or reset
    after this operation.

    References:
        [1] Bäumer, E. et al (2024). Quantum Fourier Transform Using Dynamic Circuits.
        Phys. Rev. Lett. 133, 150602.

    Args:
        qs: Qubit register to perform the QFT on.

    Returns:
        Measurement results in logical little-endian order.

    """
    bits = array(False for _ in range(n_qft))

    for logical_bit in range(n_qft):
        q_index = n_qft - logical_bit - 1

        h(qs[q_index])

        # Bit reversal.
        bits[logical_bit] = project_z(qs[q_index]).read()

        if bits[logical_bit]:
            for j in range(q_index):
                apply_phase(qs[j], pi / 2 ** (q_index - j))

    return bits


@guppy
@no_type_check
def iqft_and_measure[n_iqft: nat](qs: array[qubit, n_iqft]) -> array[bool, n_iqft]:
    r"""Dynamic inverse QFT followed by measurement.

    This implements the dynamic inverse QFT+measurement (iQFT+M) protocol described
    in [1].

    Mid-circuit measurements and classical feed-forward replace the
    controlled-phase gates of the standard unitary iQFT.
    The protocol requires one adaptive measurement/feed-forward round per qubit,
    resulting in $n$ rounds and $Theta(n^2)$ classically conditioned single-qubit phase
    corrections, rather than $\Theta(n^2)$ controlled-phase gates
    required by the standard unitary iQFT.

    This operation is not a unitary iQFT and does not produce a reusable transformed
    quantum state. All qubits in ``qs`` are measured, and the measurement results are
    returned in little-endian order. The ``qs`` register should be discarded or reset
    after this operation.

    References:
        [1] Bäumer, E. et al (2024). Quantum Fourier Transform Using Dynamic Circuits.
        Phys. Rev. Lett. 133, 150602.

    Args:
        qs: Qubit register to perform the iQFT on.

    Returns:
        Measurement results in logical little-endian order.

    """
    bits = array(False for _ in range(n_iqft))

    for i in range(n_iqft // 2):
        mem_swap(qs[i], qs[n_iqft - i - 1])

    for i in range(n_iqft):
        for j in range(i):
            if bits[j]:
                apply_phase(qs[i], -pi / 2 ** (i - j))

        h(qs[i])
        bits[i] = project_z(qs[i]).read()

    return bits
