"""Utils for making unsafe 'borrows' slightly safer."""

from typing import no_type_check

from guppylang import comptime, guppy, qubit
from guppylang.std.builtins import array, nat, owned, panic

from guppyalgos.utils.guppy.array import join_arrays, split_array


@guppy
@no_type_check
def _unsafe_array_borrow[n: nat](
    qs: array[qubit, n],
) -> array[qubit, n]:
    """'Borrow' an array of qubits.

    Removes the qubits from the input array qs, leaving it as invalid memory,
    and returning a now owned array with the same qubits.
    """
    return array(qs.take(i) for i in range(n))


@guppy
@no_type_check
def _unsafe_array_unborrow[n: nat](
    empty_target: array[qubit, n],
    borrowed_arr: array[qubit, n] @ owned,  # ty: ignore[not-subscriptable]
) -> None:
    """Undo the action of `unsafe_array_borrow`.

    `empty_target` must be the (now empty) array that was originally borrowed.
    Will error if `empty_target` is not empty, or `borrowed_arr` has had qubits
    borrowed again by something else.
    """
    for i in range(n):
        if not empty_target.is_borrowed(i):
            panic("empty_target is not empty, still contains qubits")
        if borrowed_arr.is_borrowed(i):
            panic("Cannot return borrowed element as it has been borrowed again")
        empty_target.put(borrowed_arr.take(i), i)
    borrowed_arr.discard_all_taken()


@guppy
@no_type_check
def _unsafe_array_borrow_slice[n: nat](
    arr: array[qubit, n],
    n_prefix: nat @ comptime,
    n_slice: nat @ comptime,
    n_suffix: nat @ comptime,
) -> tuple[
    array[qubit, "n_prefix"],
    array[qubit, "n_slice"],
    array[qubit, "n_suffix"],
]:
    """Temporarily splice an array into prefix, middle slice, and suffix.

    prefix = arr[:k]
    middle = arr[k:l]
    suffix = arr[l:]

    where:
        k = n_prefix
        l = n_prefix + n_slice

    The original array is fully borrowed and must eventually be restored
    with `_unsafe_array_unborrow_slice`.
    """
    _assert_three_lengths(n_prefix, n_slice, n_suffix, n)

    borrowed = _unsafe_array_borrow(arr)

    prefix, rest = split_array(borrowed, n_prefix, comptime(n_slice + n_suffix))
    middle, suffix = split_array(rest, n_slice, n_suffix)

    return prefix, middle, suffix


@guppy
@no_type_check
def _unsafe_array_unborrow_slice[
    n_prefix: nat,
    n_middle: nat,
    n_suffix: nat,
    n: nat,
](
    arr: array[qubit, n],
    prefix: array[qubit, n_prefix] @ owned,  # ty: ignore[not-subscriptable]
    middle: array[qubit, n_middle] @ owned,  # ty: ignore[not-subscriptable]
    suffix: array[qubit, n_suffix] @ owned,  # ty: ignore[not-subscriptable]
) -> None:
    """Restore a spliced array."""
    _assert_three_lengths(n_prefix, n_middle, n_suffix, n)

    rest = join_arrays(middle, suffix, comptime(n_middle + n_suffix))
    borrowed = join_arrays(prefix, rest, comptime(n))

    _unsafe_array_unborrow(arr, borrowed)


@guppy.comptime
def _assert_three_lengths(
    a: nat @ comptime,
    b: nat @ comptime,
    c: nat @ comptime,
    total: nat @ comptime,
) -> None:
    assert a + b + c == total
