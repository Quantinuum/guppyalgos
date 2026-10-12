---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  execution_mode: force
  execution_timeout: 120
---

# Library structure and philosophy

## Experimental releases

Any release before version 1.0.0 is experimental, and the API may change between
releases. Pin the package version for reproducible projects and check the
release notes before upgrading.

guppyalgos is built on the Guppy language framework. If Guppy is new to
you, start with the
[Guppy language guide](https://docs.quantinuum.com/guppy/language_guide/language_guide_index.html)
for an introduction to the language, type system, and programming model.


The source tree separates Python code that builds an algorithm from the Guppy
code that runs on quantum registers:

```text
guppyalgos/
├── algorithms/
│   ├── amplitude_amplification/
│   ├── block_encoding/
│   ├── phase_estimation/
│   ├── select/
│   ├── state_preparation/
│   └── time_evolution/
├── errors/
├── primitives/
│   ├── arithmetic/
│   ├── gate_decompositions/
│   ├── measurement/
│   ├── pauli/
│   ├── rotations/
│   ├── state_preparation/
│   └── subroutines/
├── testing/
└── utils/
    ├── guppy/
    └── python/
```

- `algorithms/` contains higher-level, user-facing constructions, including
  amplitude amplification, phase estimation, and time evolution. An algorithm
  often starts from classical data, such as a Hamiltonian, so a Python builder
  or factory processes that data and constructs the Guppy program.

- `primitives/` contains small reusable components from which algorithms are
  assembled. Most are Guppy functions parameterized by register types and
  compile-time sizes; a few require Python-side construction.
  - `qft` is one generic Guppy function that specializes to the compile-time
    width of its input register.
  - `fanout_basic` and `fanout_log` share the same typed interface, allowing an
    algorithm to choose a sequential or logarithmic-depth implementation.

- `gate_decompositions/` contains implementations or close variants of gates
  also available in Guppy's standard library; use it when an algorithm needs a
  particular decomposition rather than a standard gate invocation.
- `subroutines/` is a home for gate-like operations that act on whole registers
  but do not fit a more specific primitive category.
- `errors/` contains shared error definitions used to report invalid classical
  inputs or unsupported constructions consistently.
- `testing/` contains statevector and unitary inspection helpers shared by
  the test suite and example notebooks.
- `utils/guppy/` contains helpers usable inside Guppy programs, such as
  register utilities, while `utils/python/` contains host-side helpers for
  building and checking algorithms before compilation.

### Algorithms: build from classical data

At the user level, pass a Zixy Hamiltonian to `trotter_first_order`, then call
the resulting step from an ordinary Guppy function:

```{code-cell} ipython3
from guppylang import guppy
from guppylang.std.builtins import array
from guppylang.std.quantum import qubit
import zixy.qubit.pauli as zqp

from guppyalgos.algorithms.time_evolution.trotter import trotter_first_order


hamiltonian = zqp.RealTermSum.from_str(
    "(-0.5, Z0 X1), (-0.1, X0 Z1)"
)
trotter_step = trotter_first_order(hamiltonian, n_state_qubits=2)


@guppy
def apply_one_trotter_step(state_qreg: array[qubit, 2]) -> None:
    trotter_step(state_qreg, 0.1)
```

- `hamiltonian` is classical Python data that determines the Pauli terms in the
  constructed step.
- `trotter_step` is the returned Guppy function. It acts on the state register
  and takes a time step; each Pauli term uses the rotation
  `angle(coefficient * time_step)`.

### Primitives: specialize from Guppy types

QFT does not need a Python builder. Its register width is part of the Guppy
type:

```{code-cell} ipython3
from guppylang import guppy
from guppylang.std.builtins import array
from guppylang.std.quantum import qubit

from guppyalgos.primitives.subroutines.qft import qft


@guppy
def apply_qft(qreg: array[qubit, 4]) -> None:
    qft(qreg)
```

- One generic `qft` definition works for every compile-time width `n_qft`.
  This is *parametric polymorphism*.
- Here, Guppy infers `n_qft = 4` from `array[qubit, 4]` and builds the
  four-qubit version. This specialization is also called *monomorphization*.
- QFT has no algorithm-specific classical input; its circuit follows from the
  register type.

This split is a design principle rather than a hard rule: some algorithms are
pure Guppy, and a small number of primitives require Python-side construction.
