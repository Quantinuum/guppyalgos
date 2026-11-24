"""Array utils."""

from typing import no_type_check

from guppylang import array, comptime, guppy
from guppylang.std.builtins import nat, nothing, owned
from guppylang.std.quantum import discard_array, qubit


@guppy
@no_type_check
def discard_nested_arrays[n_arrays: nat, n_qubits: nat](
    arrays: array[array[qubit, n_qubits], n_arrays] @ owned,  # ty: ignore[not-subscriptable]
) -> None:
    """Discard every qubit in a nested array.

    Args:
        arrays: Owned, equally sized qubit arrays to discard.

    """
    for index in range(n_arrays):
        discard_array(arrays.take(index))
    arrays.discard_all_taken()


@guppy
@no_type_check
def split_array[n: nat, T](
    arr_in: array[T, n] @ owned,  # ty:ignore[not-subscriptable]
    n_a: nat @ comptime,
    n_b: nat @ comptime,
) -> tuple[array[T, "n_a"], array[T, "n_b"]]:
    """Split an owned array into partitions of length `n_a` n_b`."""
    _assert_lengths_consistent(n_a, n_b, n)

    a_out = array(nothing[T]() for _ in range(n_a))
    b_out = array(nothing[T]() for _ in range(n_b))
    for i in range(n_a):
        a_out[i].swap(some(arr_in.take(i))).unwrap_nothing()
    for i in range(n_b):
        b_out[i].swap(some(arr_in.take(i + n_a))).unwrap_nothing()
    arr_in.discard_all_taken()

    a_out = array(el.unwrap() for el in a_out)
    b_out = array(el.unwrap() for el in b_out)
    return a_out, b_out


@guppy
@no_type_check
def join_arrays[n_a: nat, n_b: nat, T](
    arr_a: array[T, n_a] @ owned,  # ty:ignore[not-subscriptable]
    arr_b: array[T, n_b] @ owned,  # ty:ignore[not-subscriptable]
    n_tot: nat @ comptime,
) -> array[T, "n_tot"]:
    """Join two owned arrays."""
    _assert_lengths_consistent(n_a, n_b, n_tot)

    out = array(nothing[T]() for _ in range(n_tot))
    for i in range(n_a):
        out[i].swap(some(arr_a.take(i))).unwrap_nothing()
    for i in range(n_b):
        out[n_a + i].swap(some(arr_b.take(i))).unwrap_nothing()
    arr_a.discard_all_taken()
    arr_b.discard_all_taken()

    return array(el.unwrap() for el in out)


@guppy
@no_type_check
def join_nested_arrays[n_per_register: nat, n_registers: nat, T](
    arrays: array[array[T, n_per_register], n_registers] @ owned,  # ty:ignore[not-subscriptable]
    n_total: nat @ comptime,
) -> array[T, "n_total"]:
    """Join nested arrays of the same size into one flat owned array."""
    _assert_nested_lengths_consistent(n_per_register, n_registers, n_total)

    out = array(nothing[T]() for _ in range(n_total))
    for register in range(n_registers):
        inner = arrays.take(register)
        for element in range(n_per_register):
            flat_index = register * n_per_register + element
            out[flat_index].swap(some(inner.take(element))).unwrap_nothing()
        inner.discard_all_taken()
    arrays.discard_all_taken()

    return array(element.unwrap() for element in out)


@guppy
@no_type_check
def split_nested_array[n_total: nat, T](
    arr_in: array[T, n_total] @ owned,  # ty:ignore[not-subscriptable]
    n_per_register: nat @ comptime,
    n_registers: nat @ comptime,
) -> array[array[T, "n_per_register"], "n_registers"]:  # ty:ignore[not-subscriptable]
    """Split a flat owned array into nested arrays of the same size."""
    _assert_nested_lengths_consistent(n_per_register, n_registers, n_total)

    nested_options = array(
        array(nothing[T]() for _ in range(n_per_register)) for _ in range(n_registers)
    )
    for register in range(n_registers):
        for element in range(n_per_register):
            flat_index = register * n_per_register + element
            nested_options[register][element].swap(
                some(arr_in.take(flat_index))
            ).unwrap_nothing()
    arr_in.discard_all_taken()

    nested = array(
        array(element.unwrap() for element in nested_options.take(register))
        for register in range(n_registers)
    )
    nested_options.discard_all_taken()
    return nested


@guppy.comptime
def _assert_lengths_consistent(
    n_a: nat @ comptime, n_b: nat @ comptime, n_tot: nat @ comptime
) -> None:
    assert n_a + n_b == n_tot


@guppy.comptime
def _assert_nested_lengths_consistent(
    n_per_register: nat @ comptime,
    n_registers: nat @ comptime,
    n_total: nat @ comptime,
) -> None:
    assert n_per_register * n_registers == n_total
