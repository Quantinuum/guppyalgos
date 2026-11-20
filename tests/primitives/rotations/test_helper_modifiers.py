"""Public modifier integration checks for the migrated circuits."""

from typing import no_type_check

from guppylang import guppy
from guppylang.std.builtins import array, control, dagger
from guppylang.std.quantum import qubit, angle

import zixy.qubit.pauli as zqp
from guppyalgos.primitives.pauli import pauli_to_z_basis
from guppyalgos.primitives.rotations import givens_rotation
from guppyalgos.primitives.rotations.register_incremented_givens_rotation import (
    xx_basis_change,
)
from guppyalgos.primitives.rotations.repeat_until_success_arbitrary import (
    dummy_theta_resource_state,
)
from guppyalgos.primitives.rotations.qrom_rotation import qrom_identity


def test_rotation_helper_compilation() -> None:
    """Compile generated modes for basis factories and runtime-angle helpers."""
    basis = pauli_to_z_basis(zqp.String.from_str("X0 Y1"), 2)

    @guppy
    @no_type_check
    def circuit(qs: array[qubit, 2], cs: array[qubit, 2], theta: angle) -> None:
        with control(cs), dagger:
            basis(qs)
            givens_rotation(qs[0], qs[1], theta)
            xx_basis_change(theta, qs[0], qs[1])
            dummy_theta_resource_state(qs[0], theta)
            qrom_identity(qs[0], qs[1])

    # Guppy 1.1.1 default normalization panics on the mixed generic/angle call.
    circuit.with_minimal_opt().compile_function()
