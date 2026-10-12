---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  execution_mode: force
  execution_timeout: 120
---

# Getting started

`guppyalgos` provides reusable building blocks for quantum programs written in
[guppy](https://docs.quantinuum.com/guppy/language_guide/language_guide_index.html). A typical workflow is:

1. Use Python to choose or construct a library component.
2. Call that component from a guppy function.
3. Type-check and compile the guppy program.

The library requires Python 3.12 or newer.

## Installation

We recommend [uv](https://docs.astral.sh/uv/) for managing Python and project
dependencies. After installing uv, create a project and add `guppyalgos`:

```console
uv init my-quantum-project
cd my-quantum-project
uv add guppyalgos
```

This creates an isolated environment and records `guppyalgos` as a project
dependency. To contribute to the library or work from a source checkout, use
the repository's locked environment instead:

```console
git clone https://github.com/Quantinuum/guppyalgos.git
cd guppyalgos
uv sync --all-extras --dev
```

## Compile your first program

This program prepares a uniform superposition on two qubits:

$$
|00\rangle
\longmapsto
\frac{1}{2}\left(|00\rangle+|01\rangle+|10\rangle+|11\rangle\right).
$$

```{code-cell} ipython3
from guppylang import guppy
from guppylang.std.quantum import discard_array

from guppyalgos.primitives.state_preparation import uniform_state
from guppyalgos.utils import qarray


uniform = uniform_state(4)


@guppy
def main() -> None:
    register = qarray(2)
    uniform(register)
    discard_array(register)


main.check()
package = main.compile()
```

- `uniform_state(4)` runs in Python and builds a guppy function for a uniform
  state over four basis states.
- `@guppy` marks `main` as code that guppy will type-check and compile.
- `qarray(2)` allocates two qubits in $|00\rangle$.
- `uniform(register)` applies the function built above. guppy infers the
  two-qubit register type from `register`.
- `discard_array(register)` consumes the qubits when they are no longer needed.
  guppy requires every qubit to be returned, measured, or discarded.
- `main.check()` checks types and qubit ownership without compiling.
- `main.compile()` produces a HUGR package for a compatible runtime or
  simulator.

## Where to go next

- Work through the {doc}`getting-started notebook <examples/core_concepts/getting_started>`
  for executable examples of circuits, higher-order functions, structs, and protocols.
- Read the {doc}`user guide <user-guide>` for registers, higher-order
  functions, structs, protocols, and larger algorithm examples.
- Browse the {doc}`example notebooks <examples_index>` for complete programs
  covering state preparation, arithmetic, Hamiltonian simulation, and phase
  estimation.
- Use the {doc}`API reference <api/api>` when you know which component you
  need and want its exact signature and options.
