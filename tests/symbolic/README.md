# Symbolic tests

`test_symbolic.py` contains tests for checking classical reversible circuits of guppyalgos on every input of each tested size. The verifier runs the HUGR of a circuit with each qubit holding a boolean function of the input bits, stored as a binary decision diagram (BDD). Then the verifier compares each output with a Python specification from `specs.py`. The file `test_verifier.py` contains tests for the verifier itself.

The verifier can check Guppy functions that:

- use only X, CX and Toffoli gates, and the AND and relative-phase CCCX gadgets;
- bring every qubit they allocate back to |0> before discarding it.

## Adding a test

Add a row to `CASES` in `test_symbolic.py`: `(component, shapes, spec, sizes, zero, ancillas)`.

- `component`: the guppy function to test, with at most 4 arguments. For more, write a test that calls `verify` directly, like `test_cnx_ladder_logdepth`.
- `shapes`: one entry per argument. `"q"` is a qubit, `"n"` an array of `n` qubits, and `"i"` an index register for `n` items. For other types, give a tuple of Guppy types with `{n}` for the size, such as `("n", "array[qubit, comptime({n} + 1)]", "array[qubit, 4]")` for arrays of `n`, `n + 1` and 4 qubits. For a size that Python computes, an entry can also be a function of `n` that returns the type, such as `lambda n: f"array[qubit, {n.bit_length()}]"` for an array of `n.bit_length()` qubits.
- `spec`: a Python function in `specs.py` that takes the value of each argument before the circuit, and returns their values after it, as a tuple. A qubit array comes as a `SymInt`, an integer that can be added, subtracted, compared and sliced (least significant bit first), and a qubit as a BDD. For example, an adder that adds `a` into `b` modulo 2^n uses `add`, and its controlled version uses `with_control(add)`.
- `sizes`: the values of `n` to check. For example, most tests use the constant `SIZES`, or `SMALL_SIZES` when the test is slow. Leave out the sizes the function doesn't support, as `SIZES[1:]` does for n = 1.
- `zero`: the arguments that start in |0> instead of holding any input, such as a clean carry. They are named `a`, `b`, `c` and `d` by position.
- `ancillas`: the most qubits the function holds at once besides its arguments, as a number or a function of `n`. Use `None` to skip this check.

A function with these argument shapes and no return value goes directly into a row. Other functions need a wrapper in `test_symbolic.py`, for example for:

- classical arguments, such as the constant `k` in `multiplier_in_place_by(k)`;
- a method of a struct, such as `method(ToffoliLadderLinear, "descending")`;
- two functions checked together, such as an adder then its dagger in `adder_then_dagger`;
- a circuit that a library factory builds from classical data, such as `hamming_weight`.
