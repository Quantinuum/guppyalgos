"""BLISS-THC Select circuit helpers."""

from __future__ import annotations

from typing import no_type_check

from guppylang import guppy
from guppylang.std.builtins import Function, array, nat
from guppylang.std.quantum import ch, cx, cz, discard, h, qubit, toffoli, x

from guppyalgos.primitives.subroutines.equality import equality_test
from guppyalgos.primitives.gate_decompositions.cnx.ccu import ccz
from guppyalgos.primitives.rotations import (
    QROMRotations,
    Rotator,
    qrom_identity,
)
from guppyalgos.algorithms.select import SelectRotator
from guppyalgos.utils import discard_nested_arrays, qarray
from guppyalgos.utils.guppy.array import join_arrays, split_array
from guppyalgos.utils.guppy.gates import _ccswap, _cswap


@guppy
@no_type_check
def _cntrl_z[n_rotation_q: nat](
    control: qubit,
    rotation_regs: array[qubit, n_rotation_q],
) -> None:
    """Apply controlled Z to the first rotation target."""
    cz(control, rotation_regs[0])


@guppy
@no_type_check
def _doubly_cntrl_z[n_rotation_q: nat](
    controls: array[qubit, 2],
    rotation_regs: array[qubit, n_rotation_q],
) -> None:
    """Apply doubly controlled Z to the first rotation target."""
    ccz(controls[0], controls[1], rotation_regs[0])


@guppy.struct
class SelectTHCCntrlRegs[n_index_q: nat]:
    """Prepared registers consumed by the BLISS-THC Select circuit.

    The circuit and register names follow Fig. 1 of Caesura et al., "Faster quantum
    chemistry simulations on a quantum computer with improved tensor factorization
    and active volume compilation", arXiv:2501.06165v1:
    https://arxiv.org/pdf/2501.06165v1.

    Args:
        one_body_flag: Flag ``c`` selecting the one-body contribution.
        coefficient_sign: Sign qubit ``m`` for the selected LCU coefficient.
        first_index_qreg: First QROM index register ``b0``.
        second_index_qreg: Second QROM index register ``b1``.

    """

    one_body_flag: qubit
    coefficient_sign: qubit
    first_index_qreg: array[qubit, n_index_q]
    second_index_qreg: array[qubit, n_index_q]


@guppy.struct
class THCWalkTargetRegs[n_modes: nat]:
    """Spin-orbital target registers acted on by the THC Select circuit.

    Args:
        spin_up: Spin-up orbital register.
        spin_down: Spin-down orbital register.

    """

    spin_up: array[qubit, n_modes]
    spin_down: array[qubit, n_modes]


@guppy.struct
class SelectTHCCntrl[
    n_index_q: nat,
    n_combined_index_q: nat,
    n_data_q: nat,
    n_givens: nat,
    n_modes: nat,
    Cascade: Rotator[
        array[array[qubit, n_data_q], n_givens],  # ty: ignore[not-subscriptable]
        array[qubit, n_modes],
    ],
]:
    """Composable BLISS-THC Select skeleton.

    The combined ``(mu, c)`` and two-body ``nu`` QROM callables are composed with
    the supplied cascade through :class:`SelectRotator`. The same cascade is moved
    through each temporary rotator in turn, allowing it to retain and reuse any
    quantum resource it owns.

    Both internal QROM rotations use identity as their uncompute operation. The
    initial QROM compute therefore leaves its angle data loaded through the central
    Pauli action, and the compute at the end of the daggered rotation clears it.

    This implements the Select register flow shown in Fig. 1 of Caesura et al.,
    "Faster quantum chemistry simulations on a quantum computer with improved tensor
    factorization and active volume compilation", arXiv:2501.06165v1:
    https://arxiv.org/pdf/2501.06165v1.

    Args:
        combined_qrom_compute: QROM loading the combined one- or two-body angles
            selected by the little-endian address ``[mu..., c]``.
        two_body_qrom_compute: QROM loading the selected two-body angles.
        cascade: Rotator applying and undoing the loaded Givens cascade.
        equality_cnx: Multi-controlled-X implementation used to compare the two
            index registers.

    """

    combined_qrom_compute: Function[
        [
            array[qubit, n_combined_index_q],
            array[array[qubit, n_data_q], n_givens],  # ty: ignore[not-subscriptable]
        ],
        None,
    ]
    two_body_qrom_compute: Function[
        [
            array[qubit, n_index_q],
            array[array[qubit, n_data_q], n_givens],  # ty: ignore[not-subscriptable]
        ],
        None,
    ]
    cascade: Cascade
    equality_cnx: Function[[array[qubit, n_index_q], qubit], None]

    @guppy
    @no_type_check
    def compose(
        self,
        control_qubit: qubit,
        prep_register: SelectTHCCntrlRegs[n_index_q],
        target_registers: THCWalkTargetRegs[n_modes],
    ) -> None:
        """Apply a BLISS-THC Select circuit matching the Fig. 1 register flow.

        Args:
            control_qubit: External control for the encoded Select operation.
            prep_register: Decoded THC indices and coefficient flags.
            target_registers: Spin-up and spin-down orbital registers.

        """
        index_swap_control = qubit()
        indices_equal = qubit()
        first_spin_swap = qubit()
        second_spin_swap = qubit()
        select_control = qubit()
        cx(control_qubit, select_control)

        h(index_swap_control)
        equality_test(
            self.equality_cnx,
            prep_register.first_index_qreg,
            prep_register.second_index_qreg,
            indices_equal,
        )

        h(first_spin_swap)

        x(indices_equal)
        ch(indices_equal, second_spin_swap)
        x(indices_equal)

        x(first_spin_swap)
        toffoli(
            indices_equal,
            first_spin_swap,
            second_spin_swap,
        )
        x(first_spin_swap)

        # Swap the indices only in the i = 1, c = 0 subspace.
        x(prep_register.one_body_flag)
        _ccswap(
            prep_register.one_body_flag,
            index_swap_control,
            prep_register.first_index_qreg,
            prep_register.second_index_qreg,
        )
        x(prep_register.one_body_flag)

        # PREPARE encodes coefficient magnitudes, so this controlled phase supplies
        # the sign (-1)^m only in the selected ``control = 1`` branch.
        cz(select_control, prep_register.coefficient_sign)

        data_qregs = array(qarray(n_data_q) for _ in range(n_givens))

        combined_index_qreg = join_arrays(
            prep_register.first_index_qreg,
            array(prep_register.one_body_flag),
            n_combined_index_q,
        )
        first_qrom_rotations = QROMRotations(
            self.combined_qrom_compute,
            self.cascade,
            qrom_identity[
                array[qubit, n_combined_index_q],
                array[array[qubit, n_data_q], n_givens],
            ],
        )
        first_select = SelectRotator(first_qrom_rotations, _cntrl_z)
        _cswap(first_spin_swap, target_registers.spin_up, target_registers.spin_down)
        first_select.compose(
            combined_index_qreg,
            data_qregs,
            select_control,
            target_registers.spin_up,
        )

        prep_register.first_index_qreg, one_body_flag_qreg = split_array(
            combined_index_qreg,
            n_index_q,
            1,
        )
        prep_register.one_body_flag = one_body_flag_qreg.take(0)
        one_body_flag_qreg.discard_all_taken()

        self.cascade = first_select.qrom_rotations.rotation_box

        cx(first_spin_swap, second_spin_swap)
        _cswap(second_spin_swap, target_registers.spin_up, target_registers.spin_down)
        cx(first_spin_swap, second_spin_swap)

        second_qrom_rotations = QROMRotations(
            self.two_body_qrom_compute,
            self.cascade,
            qrom_identity[
                array[qubit, n_index_q],
                array[array[qubit, n_data_q], n_givens],
            ],
        )
        second_select = SelectRotator(
            second_qrom_rotations,
            _doubly_cntrl_z,
        )
        # Disable the second rotated Z in the one-body sector c = 1.
        x(prep_register.one_body_flag)
        second_select_controls = array(select_control, prep_register.one_body_flag)
        second_select.compose(
            prep_register.second_index_qreg,
            data_qregs,
            second_select_controls,
            target_registers.spin_up,
        )
        self.cascade = second_select.qrom_rotations.rotation_box
        select_control = second_select_controls.take(0)
        prep_register.one_body_flag = second_select_controls.take(1)
        second_select_controls.discard_all_taken()
        x(prep_register.one_body_flag)
        _cswap(second_spin_swap, target_registers.spin_up, target_registers.spin_down)

        # Undo the index swap with the same open c control.
        x(prep_register.one_body_flag)
        _ccswap(
            prep_register.one_body_flag,
            index_swap_control,
            prep_register.first_index_qreg,
            prep_register.second_index_qreg,
        )
        x(prep_register.one_body_flag)
        x(index_swap_control)

        # Unprepare the spin-selection ancillas in reverse order.
        x(first_spin_swap)
        toffoli(
            indices_equal,
            first_spin_swap,
            second_spin_swap,
        )
        x(first_spin_swap)
        x(indices_equal)
        ch(indices_equal, second_spin_swap)
        x(indices_equal)
        h(first_spin_swap)

        discard_nested_arrays(data_qregs)
        equality_test(
            self.equality_cnx,
            prep_register.first_index_qreg,
            prep_register.second_index_qreg,
            indices_equal,
        )
        h(index_swap_control)

        discard(index_swap_control)
        discard(indices_equal)
        discard(first_spin_swap)
        discard(second_spin_swap)
        cx(control_qubit, select_control)
        discard(select_control)
