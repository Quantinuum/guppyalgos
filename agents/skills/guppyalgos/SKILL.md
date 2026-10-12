---
name: guppyalgos
description: Develop and maintain the guppyalgos Python package by using repository code, examples, tests, and docstrings as the source of truth. Use when implementing features, fixing bugs, refactoring, or adding tests in this repo.
---

# Workflow

1. Confirm repository root is the current workspace root.
2. Read in this order before editing:
   - `README.md`
   - `pyproject.toml`
   - target module under `guppyalgos/`
   - related tests under `tests/`
   - related examples under `examples/`
3. Treat tests and docstrings as the behavioral contract when explicit docs are absent.
4. Follow existing module and test patterns; avoid introducing new abstractions without clear benefit.
5. Keep edits minimal and localized.
6. Add or update tests for behavior changes.

# Fast Discovery

Run these commands from repo root:

```bash
rg --files guppyalgos tests examples
rg -n "class |def |@dataclass|@guppy|Protocol" guppyalgos tests
rg -n "TODO|FIXME|NOTE|invariant|assume" guppyalgos tests
rg -n "<feature_or_symbol_name>" guppyalgos tests examples
```

# Implementation Rules

- Preserve public signatures unless the user explicitly requests API changes.
- Prefer `ty` and type annotations for static type validation in internal code. Use runtime checks at public API boundaries or for semantic errors that would otherwise be silently accepted, misinterpreted, or produce unclear failures.
- Reuse existing naming and file organization conventions.
- Prefer extending existing modules over creating new files when feasible.
- When intent is ambiguous, infer from tests/examples and state assumptions in the final response.

## Guppy Factory Annotations

Make the Python return type annotation match the actual Guppy type returned. If
the exact type cannot be expressed, make the returned Guppy signature clear in
the docstring.

For example, use a generic target type when the returned function can accept
different register shapes:

```python
# Avoid: the target is not always a flat qubit array.
GuppyFunctionDefinition[
    [qubit, array[qubit, n_i_q], array[qubit, n_s_q]], None
]

# Prefer: this matches the returned Guppy function.
GuppyFunctionDefinition[
    [qubit, array[qubit, n_i_q], TargetRegs], None
]
```

# Validation

Run checks only for affected files first. Do not run full-suite tests unless the user asks or the change is cross-cutting.

```bash
uv run pytest tests/<target_path> -q
prek
```

When multiple modules are affected, run the union of the relevant test paths only.

If an environment dependency is unavailable (for example private indexes), report what could not be run and why.

# High-Signal Areas

- Core package: `guppyalgos/`
- Behavioral coverage: `tests/`
- Usage patterns: `examples/`
- Tooling and constraints: `pyproject.toml`

# Writing Tests

For the number of qubits typically used in tests, compilation takes longer than emulation. For long looping tests consider compiling outside
of the loop with `main.emulator(n)` and passing in the particular inputs for that test using entrypoint arguments to reduce runtime.
