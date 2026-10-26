"""Repeat-until-success Rz rotation using arbitrary resource state."""

from guppylang import guppy
from guppylang.std.builtins import output
from guppylang.std.quantum import measure, qubit, cx, rz, angle, h
from guppylang.defs import GuppyFunctionDefinition


from typing import no_type_check


@guppy(unitary=True)
@no_type_check
def dummy_theta_resource_state(q: qubit, theta: angle) -> None:
    """Prepare ``|0> + exp(-i * theta)|1>`` using Rz and H gates.

    This is a placeholder for a more efficient state preparation method.
    such as phase gradient addition etc

    Args:
        q (qubit): The qubit to prepare.
        theta (angle): The angle for the Rz rotation.

    """
    h(q)
    rz(q, -theta)


def repeat_until_success_rz(
    theta_state_method: GuppyFunctionDefinition[[qubit, float], None],
) -> GuppyFunctionDefinition[[qubit, angle], None]:
    """Repeat-until-success Rz rotation on qubit q by angle theta using ancilla qubits.

    This function applies a repeat-until-success strategy to implement
    an Rz rotation on the target qubit `q` by the specified angle `theta`.
    It uses an ancilla qubit prepared in a resource state determined by
    the `theta_state_method` function. The process is repeated until
    the measurement of the ancilla qubit indicates success. Ie 2 fauls and
    1 success for each attempt  Rz(-theta), Rz(-2theta), Rz(4theta) respectively.

    This should be used with a theta_state_method that can prepare
    arbitrary angles much more efficiently than direct Rz rotations synthesis.

    Args:
        theta_state_method (Callable[[qubit, float], None]): A function that prepares
            the ancilla qubit in the required resource state for the given angle.

    Returns:
        GuppyFunctionDefinition[[qubit, float], None]: A guppy function that performs
            the repeat-until-success Rz rotation on the target qubit `q` \
                by angle `theta`.

    """

    @guppy
    @no_type_check
    def repeat_until_success_rz_fn(
        q: qubit,
        theta: angle,
    ) -> None:
        """Guppy function to perform repeat-until-success Rz rotation.

        Args:
            q (qubit): The target qubit to apply the Rz rotation on.
            theta (angle): The angle for the Rz rotation.

        """
        attempts = 0
        while True:
            attempts += 1

            a = qubit()

            theta_state_method(a, theta * (2 ** (attempts - 1)))

            cx(q, a)

            if not measure(a).read():
                continue

            output("attempts", attempts)
            break

    return repeat_until_success_rz_fn
