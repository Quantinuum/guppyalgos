---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  execution_mode: force
  execution_timeout: 120
---

# Arithmetic

- Compute sums, comparisons, products, and powers directly on quantum registers.
- Compose the same routines with basis states or superpositions; arithmetic does
  not require measuring the inputs.
- Choose between modular results and explicit carry or borrow outputs, and use
  controlled variants inside larger algorithms.

Start with the {doc}`arithmetic demo <examples/arithmetic/arithmetic_demo>` for
runnable examples and checks. The repository default is little endian.

## Addition and subtraction

- Adders preserve `a_qreg` and update `b_qreg`.
- Modular addition keeps the low $n$ bits. A carry-out variant also records
  overflow, giving the complete sum when the carry starts at zero:

$$
|a\rangle|b\rangle|0\rangle
\longmapsto |a\rangle|(a+b)\bmod 2^n\rangle
\left|\left\lfloor\frac{a+b}{2^n}\right\rfloor\right\rangle.
$$

The snippets below share these imports and define functions to call from a
[guppy](https://docs.quantinuum.com/guppy/language_guide/language_guide_index.html)
circuit.

```{code-cell} ipython3
from guppylang import guppy
from guppylang.std.builtins import array
from guppylang.std.quantum import qubit
from guppyalgos.primitives.arithmetic import (
    adder_ripple_gidney_carry_out,
    subtractor_ripple_gidney_carry_out,
    cntrl_adder_ripple_gidney_mod,
    multiplier_ripple_gidney_mod,
    exponentiator_ripple_gidney_mod,
)


@guppy
def add(a_qreg: array[qubit, 3], b_qreg: array[qubit, 3], carry: qubit) -> None:
    adder_ripple_gidney_carry_out(a_qreg, b_qreg, carry)


@guppy
def subtract(a_qreg: array[qubit, 3], b_qreg: array[qubit, 3], borrow: qubit) -> None:
    subtractor_ripple_gidney_carry_out(a_qreg, b_qreg, borrow)
```

- With three-bit inputs $a=3$, $b=6$, addition writes $1$ into `b_qreg`
  and $1$ into a zero-initialized carry: $1+8=9$.
- Subtraction computes **$b-a$**. Its result is $(b-a)\bmod 2^n$; a
  zero-initialized borrow qubit becomes one when $b<a$.
- For $a=3$, $b=2$, subtraction gives low bits $7$ and borrow $1$,
  representing $7-8=-1$.

### Choose an adder implementation

| Family | Construction | Useful choice to explore |
| :-- | :-- | :-- |
| Cuccaro | Computes carries and reverses the carry network while writing the sum. | Configurable ladder implementations and their depth/workspace tradeoffs. |
| Gidney | Uses temporary logical ANDs with measurement-based uncomputation. | Reducing T-gate cost using workspace and measurement feedback. |

Both families provide modular and carry-out addition, subtraction, and
controlled variants. See the {doc}`addition notebook
<examples/arithmetic/ripple_carry_addition_example>` and
{doc}`Toffoli ladders <examples/gate_decompositions/toffoli_ladders>`.

## Controlled arithmetic

- A controlled adder changes the target only when its control is one.
- The control can itself be in superposition, allowing arithmetic to depend
  coherently on another part of the algorithm:

$$
|c\rangle|a\rangle|b\rangle
\longmapsto |c\rangle|a\rangle|(b+ca)\bmod 2^n\rangle.
$$

```{code-cell} ipython3
@guppy
def controlled_add(
    control: qubit, a_qreg: array[qubit, 3], b_qreg: array[qubit, 3],
) -> None:
    cntrl_adder_ripple_gidney_mod(control, a_qreg, b_qreg)
```

Controlled subtractors perform the corresponding update $b\mapsto b-ca$.

## Comparison

- Comparators preserve both inputs and XOR a comparison result into a flag.
- `comparator_vandaele(n)` builds an ancilla-free comparator for **$a<b$**.
  The demo swaps its inputs when testing $b<a$.

$$
|a\rangle|b\rangle|z\rangle
\longmapsto |a\rangle|b\rangle|z\oplus[a<b]\rangle.
$$

```{code-cell} ipython3
from guppyalgos.primitives.arithmetic.comparator import comparator_vandaele

less_than = comparator_vandaele(3)


@guppy
def compare(a_qreg: array[qubit, 3], b_qreg: array[qubit, 3], flag: qubit) -> None:
    less_than(a_qreg, b_qreg, flag)
```

- Initialize `flag` at zero to read the comparison directly.
- Use it to control another operation, then uncompute it with the same
  comparator while the inputs are unchanged.
- Cuccaro-based comparison is also available; check its argument convention
  and ladder options when choosing an implementation.

## Incrementing a register

- Incrementers implement $|x\rangle\mapsto|(x+1)\bmod 2^n\rangle$.
- `cca_incrementer` uses conditionally clean ancillas; linear-depth
  temporary-AND implementations and controlled variants are also available.

```{code-cell} ipython3
from guppyalgos.primitives.arithmetic.incrementer.incrementer_cca import cca_incrementer


@guppy
def increment(qreg: array[qubit, 4]) -> None:
    cca_incrementer(qreg)
```

For four qubits, $7\mapsto8$ and $15\mapsto0$. Register width sets the wraparound.

## Multiplication

- The Gidney multiplier composes shifted controlled additions.
- Both inputs are preserved; the product is added to an accumulator:

$$
|a\rangle|b\rangle|p\rangle
\longmapsto |a\rangle|b\rangle|(p+ab)\bmod 2^n\rangle.
$$

```{code-cell} ipython3
@guppy
def multiply(
    a_qreg: array[qubit, 3], b_qreg: array[qubit, 3], product_qreg: array[qubit, 3],
) -> None:
    multiplier_ripple_gidney_mod(a_qreg, b_qreg, product_qreg)
```

Initialize `product_qreg` at zero. For $a=3$, $b=2$, the result is $6$;
values beyond $7$ wrap modulo $8$. See the
{doc}`multiplication notebook <examples/arithmetic/multiplier>`.

## Exponentiation

- Supply a classical **odd base**, a quantum exponent, and an output register
  initialized to one.
- The circuit composes controlled multiplications by classically precomputed
  powers of the base:

$$
|x\rangle|1\rangle\longmapsto|x\rangle|b^x\bmod 2^n\rangle.
$$

```{code-cell} ipython3
@guppy
def power_of_three(exponent_qreg: array[qubit, 2], result_qreg: array[qubit, 4]) -> None:
    exponentiator_ripple_gidney_mod(exponent_qreg, result_qreg, 3)
```

For $x=3$, this produces $3^3\bmod16=11$. The base must be odd so that
multiplication is invertible modulo $2^n$. See the
{doc}`exponentiation notebook <examples/arithmetic/exponentiator>`.

The {doc}`arithmetic demo <examples/arithmetic/arithmetic_demo>` shows the
initialization, simulator qubit budgets, measurements, and numerical checks
for each of these operations.
