---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  execution_mode: force
  execution_timeout: 120
---

# Testing quantum routines

Testing is a key part of guppyalgos, even though it is not a runtime
feature. Substantial effort has gone into simulation helpers for checking
states, whole operations, and routines that depend on selected measurement
outcomes. These checks help catch errors that output bit counts alone can miss.

- **Statevectors:** did I prepare the right amplitudes and relative phases?
- **Unitaries:** does my circuit implement the expected operation on every input?
- **Post-selection:** what state or operation remains when chosen outcomes occur,
  and how likely are those outcomes?
- **Measurement paths:** do retries and corrections work after both success and
  failure?

## Set up

The examples below use the helpers in `guppyalgos.testing`. Install the
development dependencies, then save the Python snippets in a file at the
repository root and run it with `uv run python your_file.py`. Run the snippets
in order; later examples reuse earlier imports.

```console
uv sync --extra dev-dependencies
```

```{code-cell} ipython3
import numpy as np
from guppylang import guppy
from guppylang.std.builtins import array
from guppylang.std.debug import state_output
from guppylang.std.quantum import cx, discard_array, h, qubit, x, z

from guppyalgos.utils import qarray
from guppyalgos.testing import (
    assert_allclose_ignorephase,
    get_statevector,
    get_statevector_projected,
    get_unitary,
    get_unitary_projected,
)
```

## Check a prepared state

Prepare the Bell state $(|00\rangle + |11\rangle)/\sqrt{2}$ and compare its
amplitudes with the expected vector.

```{code-cell} ipython3
@guppy
def bell_state() -> None:
    state = qarray(2)
    h(state[0])
    cx(state[0], state[1])
    state_output("result_state", state)
    discard_array(state)


actual = get_statevector(bell_state, n_qubits=2)
expected = np.array([1, 0, 0, 1]) / np.sqrt(2)
assert_allclose_ignorephase(actual, expected)
```

- `state_output` records a simulation snapshot; use the tag `"result_state"`
  for `get_statevector` and record it before discarding the qubits.
- The vector contains complex amplitudes. Their squared magnitudes give
  measurement probabilities.
- `assert_allclose_ignorephase` allows one overall phase difference while still
  checking relative phases. Comparing only absolute values would miss a sign
  error between the two Bell-state terms.

## Check a whole operation

A state test checks one input. A unitary test checks the matrix whose columns
are the outputs for every computational-basis input.

```{code-cell} ipython3
@guppy
def phase_flip(state: array[qubit, 1]) -> None:
    z(state[0])


actual = get_unitary(phase_flip, n_qubits=1)
expected = np.diag([1, -1])
assert_allclose_ignorephase(actual, expected)
```

- The circuit accepts a qubit array and leaves it available to the caller.
  The helper prepares inputs and records outputs, so no `state_output` is needed.
- The minus sign matters: it changes the relative phase of a superposition,
  even though a Z gate leaves computational-basis measurement probabilities
  unchanged.
- If the circuit allocates internal work qubits, pass their count as
  `n_extra_qubits`. Return those qubits to $|0\rangle$ before discarding them.

## Check a post-selected state and its probability

Post-selection means keeping only the branch with specified measurement
outcomes. Here, the ancilla and system form a Bell pair: selecting ancilla
outcome `1` leaves the system in $|1\rangle$, with probability $1/2$.

```{code-cell} ipython3
@guppy
def heralded_state() -> None:
    ancilla = qarray(1)
    system = qarray(1)
    h(ancilla[0])
    cx(ancilla[0], system[0])
    state_output("ancilla", ancilla)
    state_output("system", system)
    discard_array(ancilla)
    discard_array(system)


selection = {"ancilla": [True]}
branch = get_statevector_projected(
    heralded_state, n_qubits=2,
    post_select_dict=selection, renormalize=False,
)
probability = np.vdot(branch, branch).real
np.testing.assert_allclose(probability, 0.5)
assert_allclose_ignorephase(branch / np.sqrt(probability), np.array([0, 1]))
```

- Dictionary keys match `state_output` tags; `False` selects `0` and `True`
  selects `1`. Each list has one entry per qubit in that register.
- Record disjoint registers together, with no computation between their
  snapshots. The helper removes selected registers and returns the remaining
  statevector.
- `renormalize=False` preserves the branch weight: its squared norm is the
  selection probability. The default, `True`, returns the normalized state
  conditioned on that outcome. Only normalize branches with nonzero probability.
- To select several registers, include each in the dictionary, for example
  `{"ancilla": [False], "phase": [False, False]}`. This is useful for checking
  the successful branch of block encodings and QSVT routines.

## Check a post-selected operation

Use `get_unitary_projected` to check a selected branch for every system input.
This example flips the system when the ancilla is `1`; selecting that outcome
extracts $X/\sqrt{2}$.

```{code-cell} ipython3
@guppy
def heralded_flip(
    ancilla: array[qubit, 1], state: array[qubit, 1],
) -> None:
    h(ancilla[0])
    cx(ancilla[0], state[0])


block = get_unitary_projected(
    heralded_flip,
    n_state_qubits=1,
    post_select_dict={"ancilla": [True]},
    pre_select_dict={"ancilla": [False]},
)
expected = np.array([[0, 1], [1, 0]]) / np.sqrt(2)
assert_allclose_ignorephase(block, expected)
```

- Projected registers come first in the circuit arguments, in dictionary order;
  the system register comes last. The helper records the projections for you.
- `pre_select_dict` specifies the initial bits of those registers. Omit it to
  initialize them all to zero.
- Despite the helper's name, the extracted block need not be unitary. Keep its
  scale: for a normalized input `psi`, `np.linalg.norm(block @ psi) ** 2` is the
  probability of the selected outcome.

## Exercise retries with predictable measurements

Selene's `QuantumReplay` supplies chosen measurement outcomes during simulation.
Use it to test paths that would otherwise depend on random shots. This small
retry loop models the success flag of a repeat-until-success routine.

```{code-cell} ipython3
from guppylang.std.builtins import output
from guppylang.std.quantum import measure
from selene_sim import QuantumReplay, Quest


@guppy
def retry_until_success() -> None:
    attempts = 0
    while True:
        flag = qubit()
        h(flag)
        attempts += 1
        if measure(flag).read():
            break
    output("attempts", attempts)


replay = QuantumReplay(
    simulator=Quest(),
    measurements=[[True], [False, False, True]],
)
results = (
    retry_until_success.emulator(1)
    .with_simulator(replay)
    .with_shots(2)
    .run()
)
```

- The first shot succeeds immediately and records one attempt. The second
  fails twice, then succeeds and records three attempts.
- Chosen outcomes must have non-negligible probability in the simulated state.
- For a real retry routine, also record and compare its final quantum state to
  check that the corrections implement the intended operation.

## Practical tips

- **Start small:** an $n$-qubit state has $2^n$ amplitudes; its full matrix has
  $4^n$ entries. Small tests can reveal mathematical errors before scaling up.
- The repository defaults to little-endian ordering.
- **Reuse compilation:** statevector helpers also accept a compiled program or
  emulator, useful when checking several inputs to the same routine.

See the {doc}`statevector testing notebook
<examples/core_concepts/statevector_testing>` for more examples, including
multiple projected registers, internal work qubits, and rotation retries.
