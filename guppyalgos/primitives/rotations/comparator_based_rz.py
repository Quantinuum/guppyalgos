"""Approximate Rz rotation with logarithmic Toffoli count."""

from math import ceil, log2
from typing import no_type_check

from guppylang import guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.angles import angle
from guppylang.std.builtins import (
    Function,
    array,
    comptime,
    exit,
    nat,
    nothing,
    output,
    some,
)
from guppylang.std.lang import Drop
from guppylang.std.option import Option
from guppylang.std.quantum import (
    cx,
    h,
    measure_array,
    qubit,
    s,
    toffoli,
    x,
    z,
)

from guppyalgos.primitives.gate_decompositions.and_op import (
    temp_and_compute,
    temp_and_uncompute,
)
from guppyalgos.utils import qarray
from guppyalgos.utils.guppy.math import floor, get_bit, tan


@guppy.protocol
class ConstantComparator[n: nat, n_ancillas: nat]:
    """A constant comparison acting on RUS and workspace registers.

    Type parameters:
        n: Number of qubits in the input register being compared.
        n_ancillas: Number of workspace qubits required by the comparator.

    """

    @guppy.require
    @no_type_check
    def compose(
        self,
        a: array[qubit, n],
        b: array[qubit, n_ancillas],
        target: qubit,
        k: int,
    ) -> None:
        """Compare the input register against the classical constant ``k``.

        Args:
            a: The ``n``-qubit input register interpreted as an integer.
            b: The ``n_ancillas``-qubit workspace register used by the
                comparator implementation.
            target: Qubit on which to accumulate the comparison result.
            k: Classical integer against which ``a`` is compared.

        """
        ...


@guppy.struct(frozen=True)
class ConstantComparatorCascade[n: nat, n_ancillas: nat]:
    """Constant comparator configured with its AND operations and direction."""

    comp_and_op: Function[[qubit, qubit, qubit], None]
    uncomp_and_op: Function[[qubit, qubit, qubit], None]
    dagger: bool

    @guppy
    @no_type_check
    def _apply_x_gates(
        self,
        a: array[qubit, n],
        target: qubit,
        k: int,
    ) -> None:
        """Apply the comparator's constant-dependent X gates."""
        for i in range(n):
            if not get_bit(k, i):
                x(a[i])

        if not get_bit(k, comptime(n - 1)):
            x(target)

    @guppy
    @no_type_check
    def _apply_toffoli_ladder(
        self,
        a: array[qubit, n],
        b: array[qubit, n_ancillas],
        target: qubit,
        k: int,
    ) -> None:
        """Apply or unapply the comparator's Toffoli ladder."""
        if self.dagger:
            toffoli(b[comptime(n - 3)], a[comptime(n - 1)], target)
            for i in range(comptime(n - 2), 1, -1):
                if get_bit(k, i + 1) != get_bit(k, i):
                    x(b[i - 1])
                self.uncomp_and_op(b[i - 2], a[i], b[i - 1])
            if get_bit(k, 2) != get_bit(k, 1):
                x(b[0])
            if get_bit(k, 0):
                self.uncomp_and_op(a[0], a[1], b[0])
            elif get_bit(k, 1):
                cx(a[1], b[0])
        else:
            if get_bit(k, 0):
                self.comp_and_op(a[0], a[1], b[0])
            elif get_bit(k, 1):
                cx(a[1], b[0])
            if get_bit(k, 2) != get_bit(k, 1):
                x(b[0])
            for i in range(2, comptime(n - 1)):
                self.comp_and_op(b[i - 2], a[i], b[i - 1])
                if get_bit(k, i + 1) != get_bit(k, i):
                    x(b[i - 1])
            toffoli(b[comptime(n - 3)], a[comptime(n - 1)], target)

    @guppy
    @no_type_check
    def compose(
        self,
        a: array[qubit, n],
        b: array[qubit, n_ancillas],
        target: qubit,
        k: int,
    ) -> None:
        """Compare ``a`` against ``k``."""
        if self.dagger:
            self._apply_toffoli_ladder(a, b, target, k)
            self._apply_x_gates(a, target, k)
        else:
            self._apply_x_gates(a, target, k)
            self._apply_toffoli_ladder(a, b, target, k)


@guppy.struct(frozen=True)
class ComparatorBasedRz[
    n: nat,
    n_ancillas: nat,
    ComparatorType: (
        ConstantComparator[  # ty: ignore[invalid-type-variable-constraints]
            n, n_ancillas
        ],
        Drop,
    ),
]:
    r"""Apply an approximate $R_z$ rotation using comparator-based RUS.

    Implements Algorithm 1 from arXiv:2404.05618, "Single-qubit rotation
    algorithm with logarithmic Toffoli count and gate depth". The algorithm
    uses a repeat-until-success approach to approximate $R_z(\theta)$ within
    error $\varepsilon$. Its success probability is greater than $1/2$.

    Algorithm:

    1. Compute $n = 1 + \lceil \log_2(1/\varepsilon) \rceil$ and
       $k = 2^{n-1} + \lfloor 2^{n-1} \tan(\theta/2) + 1/2 \rfloor$.
    2. Prepare register $a$ in superposition $|+\rangle^{\otimes n}$.
    3. Perform the comparison $a \geq k$ on the target qubit.
    4. Apply an $S$ gate to the target qubit.
    5. Apply the inverse comparison to the target qubit.
    6. Measure register $a$: if all results are zero, succeed; otherwise apply
       $Z$ and retry.

    This struct is generic over the comparator method: any concrete type that
    implements the :class:`ConstantComparator` protocol can be used. The
    comparator owns the details of the constant comparison, while
    ``n_ancillas`` captures the workspace required by that implementation. The
    :class:`ConstantComparatorCascade` implementation is the concrete cascade
    example from the paper and realizes the comparison with Clifford+Toffoli
    operations.

    Attributes:
        comparator: Configured forward constant comparator.
        inverse_comparator: Configured inverse constant comparator.
        max_attempts: Maximum number of attempts for one rotation. When it is
            reached without a success, the shot is ended with ``exit``. If
            ``nothing()``, the rotation repeats until success.

    """

    comparator: ComparatorType
    inverse_comparator: ComparatorType
    max_attempts: Option[int]

    @guppy
    @no_type_check
    def compose(self, target: qubit, theta: angle) -> None:
        """Apply an approximate Rz rotation to ``target``."""
        half_pi = 1.57079632679489661923e0
        theta_float = float(theta)
        theta_reduced = theta_float - floor(theta_float / half_pi) * half_pi
        remainder = floor(theta_float / half_pi) % 4

        if remainder > 1:
            z(target)
        if remainder & 1 == 1:
            s(target)

        if theta_reduced == 0.0:
            return

        power = 2 ** (n - 1)
        k = power + floor(float(power) * tan(theta_reduced / 2.0) + 0.5)
        attempts = 0

        while True:
            if self.max_attempts.is_some() and attempts >= self.max_attempts.unwrap():
                exit("ComparatorBasedRz reached max_attempts without success", 1)
            attempts += 1
            a_reg = qarray(n)
            b_reg = qarray(n_ancillas)

            for i in range(n):
                h(a_reg[i])

            self.comparator.compose(a_reg, b_reg, target, k)
            s(target)
            self.inverse_comparator.compose(a_reg, b_reg, target, k)
            measure_array(b_reg)

            for i in range(n):
                h(a_reg[i])

            a_measurements = measure_array(a_reg)
            all_zero = True
            for i in range(n):
                if a_measurements[i].read():
                    all_zero = False
                    break

            if all_zero:
                output("attempts", attempts)
                break
            z(target)


def comparator_based_rz_cascade(
    epsilon: float,
    max_attempts: int | None = None,
) -> GuppyFunctionDefinition[[qubit, angle], None]:
    """Build a comparator-based Rz using the cascade comparator.

    This is a convenience function that constructs a :class:`ComparatorBasedRz`
    using the :class:`ConstantComparatorCascade` implementation. The returned
    function uses temporary AND compute and uncompute operations.

    Args:
        epsilon: Approximation error bound in operator norm.
        max_attempts: Maximum number of attempts for one rotation. When it is
            reached without a success, the shot is ended with ``exit``. If
            ``None``, the rotation repeats until success.

    Returns:
        A Guppy function with signature ``(target: qubit, theta: angle) -> None``.
        The returned function uses temporary AND compute and uncompute operations.

    Raises:
        ValueError: If ``max_attempts`` is negative.

    """
    if max_attempts is not None and max_attempts < 0:
        raise ValueError("max_attempts must be non-negative")

    n = 1 + ceil(log2(1 / epsilon))
    n_comparator_ancillas = n_constant_comparator_cascade_ancillas(n)

    @guppy
    @no_type_check
    def rz_fn(target: qubit, theta: angle) -> None:
        """Apply comparator-based Rz using temporary AND operations."""
        comparator = ConstantComparatorCascade[
            comptime(n), comptime(n_comparator_ancillas)
        ](
            temp_and_compute,
            temp_and_uncompute,
            False,
        )
        inverse_comparator = ConstantComparatorCascade[
            comptime(n), comptime(n_comparator_ancillas)
        ](
            temp_and_compute,
            temp_and_uncompute,
            True,
        )
        rz = ComparatorBasedRz(
            comparator,
            inverse_comparator,
            some(comptime(max_attempts or 0))
            if comptime(max_attempts is not None)
            else nothing[int](),
        )
        rz.compose(target, theta)

    return rz_fn


def n_constant_comparator_cascade_ancillas(n: int) -> int:
    """Return the workspace qubits required by the cascade comparator."""
    return n - 2


def n_comparator_based_rz_cascade_ancillas(epsilon: float) -> int:
    """Return total RUS and cascade-comparator ancillas for ``epsilon``."""
    n = 1 + ceil(log2(1 / epsilon))
    return n + n_constant_comparator_cascade_ancillas(n)
