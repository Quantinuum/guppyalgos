"""Integers whose bits are boolean functions, to write specifications with."""


def xor(u, v):
    """Return the XOR of two boolean functions, for which ``dd`` has no operator."""
    return u.bdd.apply("xor", u, v)


class SymInt:
    """An unsigned integer whose bits are boolean functions."""

    def __init__(self, bits):
        """Store the bits, least significant first."""
        self.bits = list(bits)

    def __iter__(self):
        """Iterate over the bits."""
        return iter(self.bits)

    def __len__(self):
        """Return the number of bits."""
        return len(self.bits)

    def __getitem__(self, i):
        """Return bit ``i``, or the bits of a slice as a new integer."""
        return SymInt(self.bits[i]) if isinstance(i, slice) else self.bits[i]

    def _pair(self, other):
        """Return both operands as bit lists of the same width."""
        bdd = self.bits[0].bdd
        if isinstance(other, int):
            if other < 0:
                raise ValueError("a SymInt is unsigned")
            width = max(len(self), other.bit_length())
            other = SymInt(
                bdd.true if other >> k & 1 else bdd.false for k in range(width)
            )
        n = max(len(self), len(other))
        pad = [bdd.false] * n
        return (self.bits + pad)[:n], (other.bits + pad)[:n]

    def __add__(self, other):
        """Add, keeping the carry."""
        a, b = self._pair(other)
        carry, out = a[0].bdd.false, []
        for x, y in zip(a, b, strict=True):
            out.append(xor(xor(x, y), carry))
            carry = (x & y) | (carry & xor(x, y))
        return SymInt([*out, carry])

    def __sub__(self, other):
        """Subtract, keeping the borrow."""
        a, b = self._pair(other)
        total = SymInt(a) + SymInt(~y for y in b) + 1
        return SymInt([*total.bits[: len(a)], ~total.bits[len(a)]])

    def mul_mod(self, other):
        """Multiply modulo 2 to the number of bits of ``self``."""
        a, b = self._pair(other)
        n, zero = len(self), a[0].bdd.false
        acc = SymInt([zero] * n)
        for k, y in enumerate(b[:n]):
            acc = (acc + SymInt([zero] * k + [x & y for x in a[: n - k]]))[:n]
        return acc

    def __lt__(self, other):
        """Return the boolean function that holds where ``self < other``."""
        return (self - other).bits[-1]

    def __eq__(self, other):
        """Return the boolean function that holds where ``self == other``."""
        a, b = self._pair(other)
        equal = a[0].bdd.true
        for x, y in zip(a, b, strict=True):
            equal &= ~xor(x, y)
        return equal

    def __ne__(self, other):
        """Return the boolean function that holds where ``self != other``."""
        return ~(self == other)

    def mux(self, cond, other):
        """Select ``self`` if ``cond`` holds and ``other`` otherwise."""
        a, b = self._pair(other)
        return SymInt(cond.bdd.ite(cond, x, y) for x, y in zip(a, b, strict=True))
