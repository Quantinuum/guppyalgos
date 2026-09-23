---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  execution_mode: force
  execution_timeout: 120
---

# Upcoming features

guppyalgos is under active development. This page describes planned
directions rather than APIs that users should rely on today.

## Unified control and dagger modifiers

Guppy's built-in `control` and `dagger` modifiers already work well for simple,
straight-line unitary functions. Mark the function as `unitary` so Guppy checks
that it can be both controlled and inverted:

```{code-cell} ipython3
from guppylang import guppy
from guppylang.std.builtins import control, dagger
from guppylang.std.quantum import h, qubit, s


@guppy(unitary=True)
def basis_change(target: qubit) -> None:
    h(target)
    s(target)


@guppy
def use_modifiers(control_qubit: qubit, target: qubit) -> None:
    with control(control_qubit):
        basis_change(target)

    with dagger:
        basis_change(target)
```

- `control` adds `control_qubit` as a control to the quantum operations in
  `basis_change`.
- `dagger` reverses the operation order and replaces each gate with its
  adjoint.
- `unitary=True` declares both capabilities, and Guppy verifies the function
  body when it is defined.

The current modifier constraints do not cover every construction in this
repository:

- A `dagger` block cannot contain loops or branches.
- Allocation, measurement, reset, and discard cannot appear inside `control`
  or `dagger`.
- Functions called under `dagger` must explicitly declare the required
  capability.
- Several algorithms therefore provide separate controlled, compute, or
  uncompute implementations. The approach is correct, but it is not yet
  uniform across the library.

We are waiting for custom-modifier support in Guppy before standardizing this
across guppyalgos. The intended direction is to:

- define control and adjoint transformations through one modifier framework;
- let higher-order functions and structs request those capabilities through
  their types;
- reuse the same primitive in ordinary, controlled, and adjoint contexts where
  its implementation permits; and
- retain explicit specialized implementations where measurement-based
  uncomputation or another non-unitary technique requires them.

Until that support is available, use the controlled and inverse variants
documented by each component rather than assuming that every library function
can be placed inside `control` or `dagger`.
