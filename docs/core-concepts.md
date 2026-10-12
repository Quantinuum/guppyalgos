---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  execution_mode: force
  execution_timeout: 120
---

# Core concepts

Most code in the library is built from four ideas: quantum registers, functions,
higher-order functions, and structs.

## Quantum registers with `qarray`

A quantum register is usually an `array` of qubits. The `qarray` helper
allocates that quantum register inside a Guppy function:

$$
\operatorname{qarray}(n) \longrightarrow |0\rangle^{\otimes n}.
$$

```{code-cell} ipython3
from guppylang import guppy
from guppylang.std.quantum import discard_array

from guppyalgos.utils import qarray


@guppy
def allocate_register() -> None:
    qreg = qarray(4)
    discard_array(qreg)
```

- `qarray(4)` returns an `array[qubit, 4]` containing four fresh qubits.
- The size is known at compile time and is part of the quantum register's type.
- Qubits are linear resources: the program must use, measure, return, or
  discard every allocated qubit.

## Functions on quantum registers

Algorithm primitives are Guppy functions whose signatures state which
quantum registers they accept. A generic size allows one definition to work
for any quantum-register width:

$$
|0\rangle^{\otimes n}
\xrightarrow{\mathtt{prepare\_plus}}
H^{\otimes n}|0\rangle^{\otimes n}
=|+\rangle^{\otimes n}.
$$

```{code-cell} ipython3
from guppylang.std.builtins import array, bool, nat, owned
from guppylang.std.quantum import collect_measurements, cx, h, measure_array, qubit, rz

from guppyalgos.utils import transversal


@guppy
def prepare_plus[n: nat](qreg: array[qubit, n]) -> None:
    transversal(h, qreg)


@guppy
def measure_register[n: nat](
    qreg: array[qubit, n] @ owned,
) -> array[bool, n]:
    return collect_measurements(measure_array(qreg))
```

- `n` is inferred from the quantum register supplied by the caller.
- Guppy checks the quantum-register shape and qubit ownership at compile time.
- Function arguments are borrowed by default, as in `prepare_plus`.
- `measure_register` uses `@ owned` because measurement consumes the qubits;
  the measured quantum register cannot be used afterward.
- Repository primitives use this style for operations such as state
  preparation, arithmetic, and Hamiltonian simulation.

## Higher-order functions

A higher-order function accepts another function as a value. `Function`
records the required Guppy signature, allowing an implementation to be
replaced without changing the surrounding algorithm:

```{code-cell} ipython3
from guppylang.std.builtins import Function


@guppy
def apply_between_registers[n: nat](
    operation: Function[
        [array[qubit, n], array[qubit, n]], None
    ],
    left_qreg: array[qubit, n],
    right_qreg: array[qubit, n],
) -> None:
    operation(left_qreg, right_qreg)
```

- The injected operation receives both complete quantum registers.
- It decides how the two quantum registers interact; the wrapper only defines
  the required signature and forwards them.
- An operation with an incompatible argument or return type is rejected at
  compile time.
- Use Guppy's `Function`, rather than `typing.Callable`, for function values
  inside Guppy code.

## Generic quantum-register types

A generic quantum-register type stands for a complete quantum-register shape.
The same function can then work with arrays, tuples, or structs:

```{code-cell} ipython3
@guppy
def apply_register_operation[Regs](
    operation: Function[[Regs], None],
    qregs: Regs,
) -> None:
    operation(qregs)
```

`Regs` can be a single qubit array:

```{code-cell} ipython3
@guppy
def array_operation[n: nat](qreg: array[qubit, n]) -> None:
    transversal(h, qreg)
```

It can also be a fixed tuple. The repository uses this shape, for example,
for two-qubit Givens-rotation targets:

```{code-cell} ipython3
@guppy
def tuple_operation(qregs: tuple[qubit, qubit]) -> None:
    h(qregs[0])
    cx(qregs[0], qregs[1])
```

Starting from $|00\rangle$, this tuple operation prepares a Bell pair:

```{tikz}
:alt: A Hadamard on the first qubit followed by a controlled X to the second prepares a Bell pair from two zero qubits.

\begin{tikzcd}[column sep=0.7cm]
\lstick{$q_0: |0\rangle$} & \gate{H} & \ctrl{1} & \qw \\
\lstick{$q_1: |0\rangle$} & \qw & \targ{} & \qw
\end{tikzcd}
```

For a named bundle of several quantum registers, `Regs` can be a Guppy struct:

```{code-cell} ipython3
@guppy.struct
class WorkAndTarget[n: nat]:
    work_qreg: array[qubit, n]
    target_qreg: array[qubit, n]


@guppy
def struct_operation[n: nat](qregs: WorkAndTarget[n]) -> None:
    transversal(h, qregs.work_qreg)
    transversal(cx, qregs.work_qreg, qregs.target_qreg)
```

- The generic function does not need separate implementations for arrays,
  tuples, and structs.
- Its `Regs` type is inferred from the concrete operation and argument at the
  call site.
- Guppy checks that both agree on the entire shape.

Composed algorithms often use more than one generic quantum-register type. For
example, an algorithm can keep preparation and target quantum registers
distinct without fixing their sizes or layouts:

```{code-cell} ipython3
@guppy
def apply_select[PrepRegs, TargetRegs](
    select: Function[[PrepRegs, TargetRegs], None],
    prep_qreg: PrepRegs,
    target_qreg: TargetRegs,
) -> None:
    select(prep_qreg, target_qreg)
```

- `PrepRegs` is the type of the quantum registers used to select an operation.
- `TargetRegs` is the type acted on by that selected operation.
- The caller's concrete arguments determine both types.
- Either type can be a qubit array, tuple, or struct containing several
  quantum registers.
- Reusing the type names in the function signature guarantees that `select`
  accepts the same quantum-register shapes supplied to `apply_select`.

This separation lets one algorithm work with different quantum-register
layouts and oracle implementations. The {ref}`advanced register-composition example
<advanced-register-composition>` compares a single-register type with a
two-register bundle.

(advanced-register-composition)=
## Advanced register composition with generic types

Generic quantum-register types let a Guppy function connect component
functions without fixing the shape of every quantum-register bundle. A struct
can store those components and define their wiring once in a `compose` method,
as `LCU` does.

Start with one qubit array in each quantum-register bundle. The first box
shares `R_1` with the second, so it is both an output of `box_0` and an input
of `box_1`:

```{tikz}
:alt: Three quantum-register bundles, marked by slashes on their wires, pass through two boxes. Box 0 acts on quantum registers 0 and 1; box 1 acts on quantum registers 1 and 2.

\begin{tikzcd}[column sep=1cm]
\lstick{$R_0$} & \gate[wires=2]{\mathrm{box}_0}\qwbundle{} & \qw & \qw \\
\lstick{$R_1$} & \qwbundle{}                                  & \gate[wires=2]{\mathrm{box}_1} & \qw \\
\lstick{$R_2$} & \qw\qwbundle{}                              &                                    & \qw
\end{tikzcd}
```

```{code-cell} ipython3
from guppylang import guppy
from guppylang.std.angles import angle
from guppylang.std.builtins import Function, array, nat
from guppylang.std.quantum import qubit


@guppy.struct
class BoxComposition[Register0, Register1, Register2]:
    box_0: Function[[Register0, Register1], None]
    box_1: Function[[Register1, Register2, angle], None]

    @guppy
    def compose(
        self,
        qreg_0: Register0,
        qreg_1: Register1,
        qreg_2: Register2,
        theta: angle,
    ) -> None:
        self.box_0(qreg_0, qreg_1)
        self.box_1(qreg_1, qreg_2, theta)
```

- The struct stores two component boxes with different function signatures.
- Reusing `Register1` in both signatures defines their shared, type-checked
  boundary.
- A replacement box is compatible when its `Function` signature matches its
  position in the struct.

### One quantum register in every bundle

In this first specialization, every generic type is one qubit array:

```{code-cell} ipython3
@guppy
def bell_transversal[n: nat](
    left_qreg: array[qubit, n], shared: array[qubit, n]
) -> None:
    transversal(h, left_qreg)
    transversal(cx, left_qreg, shared)


@guppy
def rz_cx_transversal[n: nat](
    shared: array[qubit, n], right_qreg: array[qubit, n], theta: angle
) -> None:
    for i in range(len(shared)):
        rz(shared[i], theta)
    transversal(cx, shared, right_qreg)


@guppy
def compose_arrays(
    qreg_0: array[qubit, 3], qreg_1: array[qubit, 3],
    qreg_2: array[qubit, 3], theta: angle,
) -> None:
    composition = BoxComposition(bell_transversal[3], rz_cx_transversal[3])
    composition.compose(qreg_0, qreg_1, qreg_2, theta)


compose_arrays.check()
```

- Here, `Register0`, `Register1`, and `Register2` are all inferred as
  `array[qubit, 3]`.
- The explicit `[3]` specializes each generic box before it is stored as a
  function value.

### Two registers in the middle bundle

A generic type can instead be a struct containing several quantum registers. The
middle bundle can therefore contain two arrays without changing
`BoxComposition`:

```{code-cell} ipython3
@guppy.struct
class MiddleBundle[n: nat]:
    upper_qreg: array[qubit, n]
    lower_qreg: array[qubit, n]


@guppy
def bundled_box_0[n: nat](
    left_qreg: array[qubit, n], middle: MiddleBundle[n]
) -> None:
    transversal(h, left_qreg)
    transversal(cx, left_qreg, middle.upper_qreg)
    transversal(cx, left_qreg, middle.lower_qreg)


@guppy
def bundled_box_1[n: nat](
    middle: MiddleBundle[n], right_qreg: array[qubit, n], theta: angle
) -> None:
    for i in range(n):
        rz(middle.upper_qreg[i], theta)
        rz(middle.lower_qreg[i], theta)
    transversal(cx, middle.upper_qreg, right_qreg)
    transversal(cx, middle.lower_qreg, right_qreg)
```

```{tikz}
:alt: Quantum-register wires are marked by slashes. Register 1 is a bundle of two quantum registers. Box 0 acts on quantum register 0 and both quantum registers in the middle bundle; box 1 acts on both quantum registers in the middle bundle and quantum register 2.

\begin{tikzcd}[column sep=1cm]
\lstick{$R_0$}                & \gate[wires=3]{\mathrm{box}_0}\qwbundle{} & \qw                                  & \qw \\
\lstick{$R_1.\mathrm{upper}$} & \qwbundle{}                                  & \gate[wires=3]{\mathrm{box}_1} & \qw \\
\lstick{$R_1.\mathrm{lower}$} & \qwbundle{}                                  &                                      & \qw \\
\lstick{$R_2$}                & \qw\qwbundle{}                              &                                      & \qw
\end{tikzcd}
```

The struct is instantiated with boxes that accept the complete middle bundle:

```{code-cell} ipython3
@guppy
def compose_bundles(
    qreg_0: array[qubit, 3], middle: MiddleBundle[3],
    qreg_2: array[qubit, 3], theta: angle,
) -> None:
    composition = BoxComposition(bundled_box_0[3], bundled_box_1[3])
    composition.compose(qreg_0, middle, qreg_2, theta)


compose_bundles.check()
```

- `Register1` is now inferred as `MiddleBundle[3]`; the outer composer is
  unchanged.
- Both boxes can access `middle.upper` and `middle.lower`.
- Passing a box that expects one array alongside a box that expects
  `MiddleBundle[3]` fails at compile time because their `Register1` types do
  not match.
- Guppy does not support higher-rank polymorphic function values, so the boxes
  are specialized before being stored.

See the {doc}`abstract construction notebook
<examples/core_concepts/abstract_construction>`
for both complete executable specializations.

## Structs for composed algorithms

A higher-order function works well for one small injected operation. Use a
Guppy struct when several oracles must be stored together, their quantum-register
wiring is more involved, or construction should be separate from use.

For example, the repository's `LCU` struct stores its three oracles when it is
initialized, then exposes the block encoding through `compose`:

$$
U_A = \mathrm{PREPARE}^{\dagger}\,
\mathrm{SELECT}\,\mathrm{PREPARE},
\qquad
(\langle 0^a|\otimes I)U_A(|0^a\rangle\otimes I)=\frac{A}{\alpha}.
$$

```{code-cell} ipython3
@guppy.struct
class LCU[PrepRegs, TargetRegs]:
    prepare: Function[[PrepRegs], None]
    select: Function[[PrepRegs, TargetRegs], None]
    unprepare: Function[[PrepRegs], None]

    @guppy
    def compose(self, prep_qreg: PrepRegs, target_qreg: TargetRegs) -> None:
        self.prepare(prep_qreg)
        self.select(prep_qreg, target_qreg)
        self.unprepare(prep_qreg)
```

```{tikz}
:alt: An LCU block encoding. PREPARE creates a coefficient state on the preparation register, SELECT applies an operation to the target register controlled by that state, and PREPARE dagger uncomputes the preparation register. Slashes mark register bundles rather than individual qubits.

\begin{tikzcd}[column sep=0.8cm]
\lstick{$|0\rangle_{\mathrm{prep}}$}
  & \gate{\mathrm{PREPARE}}\qwbundle{}
  & \gate[wires=2]{\mathrm{SELECT}}
  & \gate{\mathrm{PREPARE}^{\dagger}}
  & \qw
  & \qw \\
\lstick{$|\psi\rangle_{\mathrm{target}}$}
  & \qw\qwbundle{}
  &
  & \qw
  & \qw
  & \qw
\end{tikzcd}
```

Construct the composed oracle once, then call it where it is needed:

```{code-cell} ipython3
@guppy
def apply_lcu[PrepRegs, TargetRegs](
    prepare: Function[[PrepRegs], None],
    select: Function[[PrepRegs, TargetRegs], None],
    unprepare: Function[[PrepRegs], None],
    prep_qreg: PrepRegs, target_qreg: TargetRegs,
) -> None:
    lcu = LCU(prepare, select, unprepare)
    lcu.compose(prep_qreg, target_qreg)
```

- The struct stores the interchangeable component boxes.
- `compose` defines their order and quantum-register wiring once, and can be called
  wherever the initialized oracle is needed.
- Reusing `PrepRegs` across all three signatures guarantees that they agree
  on the preparation-register shape.
- A generic quantum-register type may be one qubit array or a struct containing
  several quantum registers.

The repository uses this pattern for compositions including LCU block
encodings, qubitization, QSVT, reflections, and rotation strategies.

## Protocols for interchangeable structs

A protocol defines the methods a component must provide without requiring a
particular struct. Here, a compatible component must apply an operation
between two equal-sized quantum registers:

```{code-cell} ipython3
@guppy.protocol
class TwoRegisterOperation[n: nat]:
    @guppy.require
    def apply(
        self,
        left_qreg: array[qubit, n],
        right_qreg: array[qubit, n],
    ) -> None:
        ...
```

A concrete struct satisfies the protocol by providing `apply` with the same
signature. Here it stores a function that acts on both complete quantum
registers:

```{code-cell} ipython3
@guppy
def entangle_registers[n: nat](
    left_qreg: array[qubit, n],
    right_qreg: array[qubit, n],
) -> None:
    transversal(cx, left_qreg, right_qreg)


@guppy.struct
class TwoRegisterLayer[n: nat]:
    operation: Function[
        [array[qubit, n], array[qubit, n]], None
    ]

    @guppy
    def apply(
        self,
        left_qreg: array[qubit, n],
        right_qreg: array[qubit, n],
    ) -> None:
        self.operation(left_qreg, right_qreg)
```

The component is passed into the algorithm when its struct is initialized.
The algorithm can then expose its own `compose` method without requiring that
name from every component:

```{code-cell} ipython3
@guppy.struct
class RegisterAlgorithm[
    n: nat,
    Operation: TwoRegisterOperation[n],
]:
    operation: Operation

    @guppy
    def compose(
        self,
        left_qreg: array[qubit, n],
        right_qreg: array[qubit, n],
    ) -> None:
        self.operation.apply(left_qreg, right_qreg)


@guppy
def apply_cx_layer() -> None:
    left_qreg = qarray(4)
    right_qreg = qarray(4)

    algorithm = RegisterAlgorithm(
        TwoRegisterLayer[4](entangle_registers[4])
    )
    algorithm.compose(left_qreg, right_qreg)

    discard_array(left_qreg)
    discard_array(right_qreg)
```

- `TwoRegisterOperation` enforces the component's two-register `apply`
  interface.
- `TwoRegisterLayer` satisfies the protocol structurally; it does not need to
  inherit from it.
- The stored operation receives both complete quantum registers and defines how they
  interact. Here, `entangle_registers` applies `cx` pair by pair.
- A different struct can be passed to `RegisterAlgorithm` if it provides the
  same `apply` method.
- The elementary example uses explicit array types. The {ref}`advanced
  register-composition example <advanced-register-composition>` shows how
  generic register bundles extend this pattern.
