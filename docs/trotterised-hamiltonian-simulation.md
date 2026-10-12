---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  execution_mode: force
  execution_timeout: 120
---

# Trotterized Hamiltonian simulation

Trotterization approximates Hamiltonian time evolution by applying
exponentials of simpler terms:

$
H=\sum_{j=1}^{m}h_j,
\qquad
U(t)=e^{-iHt}.
$

## Zixy Hamiltonians

[Zixy](https://github.com/CQCL/zixy) stores a Hamiltonian as a sum of
coefficient-weighted Pauli strings:

$
H=\sum_{j=1}^{m}a_jP_j,
\qquad
P_j\in\{I,X,Y,Z\}^{\otimes n}.
$

```{code-cell} ipython3
import zixy.qubit.pauli as zqp

pauli_string = zqp.String.from_str("Z0 X1", 2)
hamiltonian = zqp.RealTermSum.from_str(
    "(-0.5, Z0 X1), (-0.1, X0 Z1), (-0.2, Y0 Y1)"
)
```

- `zqp.String` represents one tensor product of Pauli operators.
- `zqp.RealTermSum` represents a Hermitian Pauli Hamiltonian by pairing those
  strings with real coefficients.
- `trotter_first_order` converts every non-identity term into a Pauli
  exponential.

## Pauli exponentials

`pauli_exp` constructs a Guppy function implementing
a Pauli-string exponential:

$
U_P(\theta)=e^{-i\theta P/2}.
$

```{code-cell} ipython3
from guppyalgos.primitives.pauli.pauli_exp import cntrl_pauli_exp, pauli_exp

pauli_gadget = pauli_exp(pauli_string, n_qubits=2)
controlled_pauli_gadget = cntrl_pauli_exp(pauli_string, n_qubits=2)
```

The input may contain any tensor product of $I$, $X$, $Y$, and $Z$. Only its
non-identity support participates in the parity ladder:

| Pauli | Basis change before the parity ladder |
| --- | --- |
| $I$ | None; this qubit is omitted from the ladder. |
| $X$ | $H$ maps $X$ to $Z$. |
| $Y$ | Apply $S^\dagger$, then $H$, to map $Y$ to $Z$. |
| $Z$ | None; it is already in the required basis. |

For $P = Z_0X_1$, the ordinary gadget is:

```{tikz}
:alt: A Pauli gadget for Z on q_0 and X on q_1. A Hadamard changes q_1 to the Z basis, a pair of controlled-NOT gates surrounds an RZ rotation on q_1, and a final Hadamard restores the original basis.

\begin{tikzcd}[column sep=0.7cm]
\lstick{$q_0: Z$} & \qw      & \ctrl{1} & \qw                  & \ctrl{1} & \qw      & \qw \\
\lstick{$q_1: X$} & \gate{H} & \targ{}  & \gate{R_Z(\theta)} & \targ{}  & \gate{H} & \qw
\end{tikzcd}
```

The controlled form uses the same basis changes and parity ladder, but the
external qubit controls the central rotation:

$
C(U_P)=|0\rangle\!\langle 0|\otimes I
+|1\rangle\!\langle 1|\otimes U_P(\theta).
$

```{tikz}
:alt: A controlled Pauli gadget for Z on q_0 and X on q_1. The parity ladder surrounds an RZ rotation on q_1 controlled by an external control qubit.

\begin{tikzcd}[column sep=0.7cm]
\lstick{$c$}      & \qw      & \qw      & \ctrl{2}              & \qw      & \qw      & \qw \\
\lstick{$q_0: Z$} & \qw      & \ctrl{1} & \qw                   & \ctrl{1} & \qw      & \qw \\
\lstick{$q_1: X$} & \gate{H} & \targ{}  & \gate{R_Z(\theta)}  & \targ{}  & \gate{H} & \qw
\end{tikzcd}
```

- Basis changes map every non-identity Pauli to the Z basis.
- A CX ladder computes the joint parity onto its target qubit.
- `pauli_exp` applies `R_Z` to the parity target.
- `cntrl_pauli_exp` replaces it with `CR_Z`, leaving the target operation
  inactive when the external control is zero.
- The CX ladder and basis changes are then uncomputed.
- The uncontrolled factory rejects an identity string. The controlled factory
  preserves its observable relative phase by rotating the control qubit.

### Choose the circuit construction

The Pauli string fixes the operator, but it does not fix the circuit used for
the parity computation or terminal rotation. These inputs can be selected when
the gadget is built:

| Input | Default | Other supported use |
| --- | --- | --- |
| `pauli_string` | Required `zqp.String` | Any $I/X/Y/Z$ tensor product; `pauli_exp` requires at least one non-identity Pauli. |
| `n_qubits` | Required integer | May be larger than the Pauli support, provided every Pauli index is in range. |
| `cx_ladder` | `CXLadderLog` | `CXLadderLinear`, or another implementation of the `Ladder` protocol. |
| `rz_method` | `rz` | Any compatible `(qubit, angle) -> None` guppy function, including an RUS $R_Z$ construction. |
| `controlled_rz_method` | `crz` | For `cntrl_pauli_exp`, any compatible `(control, target, angle) -> None` guppy function. |

For example, this keeps the same $e^{-i\theta P/2}$ operation while choosing a
linear CX ladder and a repeat-until-success rotation:

```{code-cell} ipython3
from guppyalgos.primitives.rotations import (
    dummy_theta_resource_state, repeat_until_success_rz,
)
from guppyalgos.primitives.subroutines.ladders import CXLadderLinear

rus_rz = repeat_until_success_rz(dummy_theta_resource_state)
rus_pauli_gadget = pauli_exp(
    pauli_string,
    n_qubits=2,
    cx_ladder=CXLadderLinear,
    rz_method=rus_rz,
)
```

The RUS implementation changes how the central $R_Z$ is synthesized. The
basis changes, parity ladder, angle convention, and resulting Pauli
exponential remain the same.

See the {doc}`Pauli-exponential notebook
<examples/pauli_exponential/pauli_exponential>` for a
complete executable example and alternative rotation implementations.

## Constructing the Trotter simulation

A first-order Trotter simulation applies the Zixy term exponentials
sequentially, then repeats the resulting step:

$
e^{-iHt}
\approx
\left(\prod_{j=1}^{m}e^{-ih_jt/r}\right)^r,
\qquad
\text{error}=O\!\left(\frac{t^2}{r}\right).
$

```{code-cell} ipython3
from guppyalgos.algorithms.time_evolution.trotter import ham_sim_trotter, trotter_first_order

n_state_qubits = len(hamiltonian.qubits)
trotter_step = trotter_first_order(hamiltonian, n_state_qubits)
simulation = ham_sim_trotter(
    trotter_step, n_steps=10, time_step=0.01,
    n_state_qubits=n_state_qubits,
)
```

The Hamiltonian and product-formula order determine the sequence of Pauli
exponentials. The ladder and rotation inputs are passed to every term, so one
choice changes the complete Trotter step consistently.

| Builder | Input Hamiltonian or schedule | Result |
| --- | --- | --- |
| `trotter_first_order` | `zqp.RealTermSum` | One forward pass through the non-identity terms. |
| `cntrl_trotter_first_order` | `zqp.RealTermSum` | Controlled forward pass; retains identity terms as relative phases. |
| `trotter_higher_order` | `zqp.RealTermSum` and even `order >= 2` | Symmetric Suzuki formula; higher orders recursively reduce product-formula error. |
| `cntrl_trotter_higher_order` | Same inputs plus controlled rotation methods | Controlled symmetric Suzuki formula. |
| `trotter_from_sequence` | Terms and `(term_index, time_factor)` pairs | Custom ordering, repeated terms, and signed time factors. |

For example, a second-order step applies half steps forward and backward,

$$
S_2(t)=
\prod_{j=1}^{m}e^{-i h_jt/2}
\prod_{j=m}^{1}e^{-i h_jt/2},
$$

and is built with:

```{code-cell} ipython3
from guppyalgos.algorithms.time_evolution.trotter import trotter_higher_order

second_order_step = trotter_higher_order(
    hamiltonian,
    n_state_qubits,
    order=2,
    cx_ladder=CXLadderLinear,
)
```

At runtime, each term receives an angle equal to its coefficient multiplied by
the sequence factor and `time_step`. Negative sequence factors implement
backward evolution. `ham_sim_trotter` then repeats the completed step
`n_steps` times.

See the {doc}`Trotter Hamiltonian-simulation notebook
<examples/hamiltonian_simulation/ham_sim_trotter_demo>` for the complete
construction, execution, and
accuracy comparison.

Continue with {doc}`phase-estimation` to use controlled Trotter steps in QPE.
