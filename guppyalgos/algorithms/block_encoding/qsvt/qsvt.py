"""QSVT module."""

from typing import no_type_check

from guppylang import guppy
from guppylang.std.quantum import qubit
from guppylang.std.builtins import array, frozenarray, nat
from guppylang.std.quantum import h, x, rz
from guppylang.std.angles import angle

from guppyalgos.primitives.gate_decompositions.cnx.cnx import cnx
from guppyalgos.algorithms.block_encoding.lcu import LCU


@guppy.struct
@no_type_check
class QSVT[
    n_prep: nat,
    TargetRegs,
    n_phases: nat,
]:
    """Guppy struct representing the QSVT algorithm.

    This function implements the QSVT algorithm as described in
    https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.040203, following
    the convention used in Appendix A2 of the paper. Given a function for the desired
    LCU it composes it with QSP phases. the LCU is projected on to the signal register
    and polynomial transform is performed on the LCU block encoded matrix.

    The tranfromed matrix is then obtained by postselecting on the ancilla signal qubit
    and the prepare qubits.

    Attributes:
        lcu: Guppy struct performing the LCU.
        lcu_dagger: Guppy struct performing the LCU dagger.
        phases: List of QSP phases in the reflection convention.

    """

    lcu: LCU[array[qubit, n_prep], TargetRegs]
    lcu_dagger: LCU[array[qubit, n_prep], TargetRegs]
    phases: frozenarray[float, n_phases]

    @guppy
    @no_type_check
    def compose(
        self, signal: qubit, prep_qreg: array[qubit, n_prep], select_qreg: TargetRegs
    ) -> None:
        r"""Apply the QSVT.

        Args:
            signal: The signal qubit for QSP phases.
            prep_qreg: The PREPARE register.
            select_qreg: The SELECT register.

        """
        h(signal)
        for i in range(n_phases):
            if i % 2 == 0:
                self.lcu.compose(prep_qreg, select_qreg)
            else:
                self.lcu_dagger.compose(prep_qreg, select_qreg)
            _add_qsp_phase(prep_qreg, signal, self.phases[i])
        h(signal)


@guppy
@no_type_check
def _add_qsp_phase[n_prep: nat](
    prep_qreg: array[qubit, n_prep], signal: qubit, phase: float
) -> None:
    """Auxiliary function for QSVT algorithm."""
    for i in range(n_prep):
        x(prep_qreg[i])
    cnx(prep_qreg, signal)
    rz(signal, angle(phase))
    cnx(prep_qreg, signal)
    for i in range(n_prep):
        x(prep_qreg[i])
