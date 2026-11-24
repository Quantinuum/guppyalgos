"""Python specifications of the circuits that the symbolic tests check."""

import random
from functools import partial, reduce
from operator import and_

from tests.symbolic.symint import SymInt, xor


def if_else(c, u, v):
    """Return ``u`` where ``c`` holds and ``v`` elsewhere, for bits or registers."""
    return u.mux(c, v) if isinstance(u, SymInt) else c.bdd.ite(c, u, v)


def with_control(spec):
    """Return a spec that applies ``spec`` only when its leading control is set."""

    def controlled(c, *regs, **kwargs):
        return c, *map(partial(if_else, c), spec(*regs, **kwargs), regs)

    return controlled


def majority(a, b, c):
    """Return the boolean function that holds when at least two input bits are set."""
    return (a & b) | (a & c) | (b & c)


def half_add(a, b, c):
    """Add the bit ``a`` into ``b``, with the carry in ``c``."""
    return a, xor(a, b), xor(c, a & b)


def half_add_inverse(a, s, c):
    """Undo ``half_add``, where ``s`` holds the sum."""
    b = xor(a, s)
    return a, b, xor(c, a & b)


def full_add(c, a, b, d):
    """Add the bits ``c`` and ``a`` into ``b``, with the carry in ``d``."""
    return c, a, xor(xor(c, a), b), xor(d, majority(c, a, b))


def full_add_inverse(c, a, s, d):
    """Undo ``full_add``, where ``s`` holds the sum."""
    b = xor(xor(c, a), s)
    return c, a, b, xor(d, majority(c, a, b))


def add(a, b):
    """Add ``a`` into ``b``."""
    return a, (a + b)[: len(a)]


def add_carry(a, b, c):
    """Add ``a`` into ``b``, writing the overflow into initially zero ``c``."""
    return a, (a + b)[: len(a)], (a + b)[len(a)]


def sub(a, b):
    """Subtract ``a`` from ``b``."""
    return a, (b - a)[: len(a)]


def sub_carry(a, b, c):
    """Subtract ``a`` from ``b``, writing the borrow into initially zero ``c``."""
    return a, (b - a)[: len(a)], (b - a)[len(a)]


def same(*regs):
    """Return every argument unchanged."""
    return regs


def add_product(a, b, p):
    """Add the product of ``a`` and ``b`` into ``p``."""
    return a, b, (p + a.mul_mod(b))[: len(a)]


def mul_const(a, k):
    """Multiply ``a`` by the constant ``k``."""
    return (a.mul_mod(k),)


def multiply_by_power(e, a, base):
    """Multiply ``a`` by the constant ``base`` to the power ``e``."""
    if not a.bits:
        return e, a
    for bit in e:
        a = a.mul_mod(base).mux(bit, a)
        base = base**2 % 2 ** len(a)
    return e, a


def xor_hamming_weight(m, w):
    """XOR into ``w`` the number of qubits of ``m`` that are set."""
    count = reduce(SymInt.__add__, (SymInt([bit]) for bit in m))
    return m, SymInt(map(xor, w, count))


def less(a, b, t):
    """Flip ``t`` if ``a < b``."""
    return a, b, xor(t, a < b)


def inc(q):
    """Add 1 to ``q``."""
    return ((q + 1)[: len(q)],)


def dec(q):
    """Subtract 1 from ``q``."""
    return ((q - 1)[: len(q)],)


def on_low_bits(spec):
    """Restrict ``spec`` to the bits needed for bound ``n`` on an n-qubit register."""

    def restricted(*regs):
        *rest, q = regs
        k = (len(q) + 1).bit_length()
        *out, low = spec(*rest, q[:k])
        return *out, SymInt([*low, *q[k:]])

    return restricted


def multi_controlled_x(c, t):
    """Flip ``t`` if every qubit of ``c`` is set."""
    return c, xor(t, reduce(and_, c))


def multi_controlled_x_dirty(c, t, x, y):
    """Flip ``t`` if every qubit of ``c`` is set, leaving the borrowed qubits as is."""
    return c, xor(t, reduce(and_, c)), x, y


def equal(lhs, rhs, flag):
    """Flip ``flag`` if ``lhs`` and ``rhs`` are equal."""
    return lhs, rhs, xor(flag, lhs == rhs)


def fanout(c, q):
    """Flip every qubit of ``q`` if ``c`` is set."""
    return c, SymInt(xor(x, c) for x in q)


def fanout_mask(n):
    """Select the positions of an n-qubit register not 0 mod 3, so none at n = 1."""
    return [k % 3 != 0 for k in range(n)]


def masked_fanout(c, q):
    """Flip the qubits of ``q`` that ``fanout_mask`` selects if ``c`` is set."""
    return c, SymInt(
        xor(x, c) if m else x for x, m in zip(q, fanout_mask(len(q)), strict=True)
    )


def toffoli_v_chain(a, b, t):
    """Flip ``t`` by the last link of a chain of Toffolis running up ``a``."""
    link = a[0]
    for i in range(1, len(a)):
        link = xor(a[i], link & b[i - 1])
    return a, b, xor(t, link & b[-1])


def cswap(c, a, b):
    """Swap the bits or registers ``a`` and ``b`` if ``c`` is set."""
    return if_else(c, b, a), if_else(c, a, b)


def swap_if_both(c, d, a, b):
    """Swap the qubits or registers ``a`` and ``b`` if ``c`` and ``d`` are set."""
    return c, d, *cswap(c & d, a, b)


def swap_up(index, state):
    """Move the indexed item to the front, following SwapUp's swap schedule."""
    s = list(state)
    for bit in reversed(range(len(index))):
        stride = 2**bit
        for lower in range(min(stride, len(s) - stride)):
            upper = lower + stride
            s[lower], s[upper] = cswap(index[bit], s[lower], s[upper])
    for i, x in enumerate(state):  # what the swaps must do for an index in range
        s[0] = if_else(index == i, x, s[0])
    return index, s


def decode(index, out):
    """Flip ``out[i]`` if ``index`` holds ``i``."""
    return index, SymInt(xor(x, index == i) for i, x in enumerate(out))


def qrom_table(n):
    """Return up to n + 1 distinct nonzero n-bit rows, deterministically seeded by n."""
    rows = min(n + 1, 2 ** (n - 1).bit_length())
    values = random.Random(n).sample(range(1, 2**n), rows)
    return [[v >> k & 1 == 1 for k in range(n)] for v in values]


def lookup(index, target):
    """XOR row ``index`` of ``qrom_table`` into ``target`` if it exists."""
    for i, row in enumerate(qrom_table(len(target))):
        target = SymInt(
            xor(t, index == i) if b else t for t, b in zip(target, row, strict=True)
        )
    return index, target


def selectswap_entries(n, width):
    """Return seeded ``width``-bit entries, leaving SelectSWAP's last row short."""
    rows = min(n + 1, 2 ** (n - 1).bit_length())
    rng = random.Random(n)
    return [[rng.random() < 0.5 for _ in range(width)] for _ in range(n * rows - 1)]


def select_swap(i, j, regs):
    """Load row ``i`` of ``selectswap_entries`` by XOR, then swap register ``j`` up."""
    regs = list(regs)
    for e, entry in enumerate(selectswap_entries(len(regs), len(regs[0]))):
        row, r = divmod(e, len(regs))
        regs[r] = SymInt(
            xor(t, i == row) if b else t for t, b in zip(regs[r], entry, strict=True)
        )
    return i, *swap_up(j, regs)


def flip_prefix(index, state):
    """Flip ``state[k]`` for every ``k`` below ``index``, and nothing out of range."""
    return index, SymInt(
        xor(s, ~(index < k + 1) & (index < len(state))) for k, s in enumerate(state)
    )


def cx_ladder(q, descending=False, dagger=False):
    """Apply a CX from each qubit of ``q`` to the next one."""
    bits = q.bits[::-1] if descending else list(q.bits)
    steps = range(len(bits) - 1)
    for i in reversed(steps) if dagger else steps:
        bits[i + 1] = xor(bits[i + 1], bits[i])
    return (SymInt(bits[::-1] if descending else bits),)


def toffoli_ladder(q, descending=False, dagger=False):
    """Apply a Toffoli from each pair of qubits of ``q`` to the qubit after it."""
    bits = q.bits[::-1] if descending else list(q.bits)
    steps = range(0, len(bits) - 2, 2)
    for i in reversed(steps) if dagger else steps:
        bits[i + 2] = xor(bits[i + 2], bits[i] & bits[i + 1])
    return (SymInt(bits[::-1] if descending else bits),)


def cnx_ladder(a, b, t, x, y, ancillae, inverse=False):
    """Apply the C^nX ladder ending at ``t``, reversed when ``inverse`` is set."""
    wires = [*a, t]
    steps = range(len(a))
    for i in reversed(steps) if inverse else steps:
        wires[i + 1] = xor(wires[i + 1], reduce(and_, b[i], wires[i]))
    return SymInt(wires[:-1]), b, wires[-1], x, y, ancillae
