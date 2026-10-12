---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  execution_mode: force
  execution_timeout: 120
---

# Measurement

- Prepare a state, choose a Pauli observable, and sample its measurement circuit.
- Use the library estimators to calculate expectation values and standard errors.

## Direct Pauli measurements

- `make_direct_measure_pauli_simple` constructs basis changes and full-register
  measurement for a Pauli string, including products such as $X_0Z_1$.
- `estimate_pauli_observable_expectation_from_bitstrings` converts the measured
  bitstrings into the relevant parity and combines weighted terms.
- See the {doc}`direct measurement notebook
  <examples/measurement/operator_averaging_direct>` for a complete circuit.

## Hadamard-test measurements

- `make_hadamard_test_pauli` measures a Pauli expectation through an ancilla.
- `estimate_expectation_from_binary_samples` handles its binary readout.
- For a sum of terms, use
  `estimate_pauli_observable_expectation_from_binary_samples` with a separate
  sample collection for each term.
- See the {doc}`Hadamard-test notebook <examples/measurement/operator_averaging>`.

## Expectation values and uncertainty

For a binary Pauli measurement, `False` represents $+1$ and `True` represents
$-1$. The estimator returns the sample mean and its estimated standard error:

$$
\widehat{\langle P\rangle}=\frac{N_+-N_-}{N},
\qquad
\mathrm{SE}=\sqrt{\frac{1-\widehat{\langle P\rangle}^2}{N}}.
$$

```{code-cell} ipython3
from guppyalgos.primitives.measurement import estimate_expectation_from_binary_samples

estimate = estimate_expectation_from_binary_samples({False: 750, True: 250})
print(f"Expectation: {estimate.expectation:.3f}")
print(f"Standard error: {estimate.standard_error:.3f}")
```

For a real observable $H=\sum_j c_jP_j$, the observable estimators combine
independently sampled terms:

$$
\widehat{\langle H\rangle}=\sum_j c_j\widehat{\langle P_j\rangle},
\qquad
\mathrm{SE}(H)^2=\sum_j c_j^2\mathrm{SE}(P_j)^2.
$$

- Sample each term in its corresponding basis; computational-basis shots alone
  do not estimate arbitrary X or Y observables.
- This uncertainty calculation assumes independent term samples. Grouped
  measurements require covariance terms.
- The {doc}`Trotter demo
  <examples/hamiltonian_simulation/ham_sim_trotter_demo>` uses these routines to
  plot measured dynamics with error bars.

See all {doc}`measurement notebooks <example_indexes/measurement_index>`.
