In addition to these review instructions, also view the instructions in ./AGENTS.md and ./agents/skills for review guidance.

## Style Guide / Review Instructions

- Bitstrings represented in `array[bool, n]` or `array[qubit, n]` should be little endian, and functions should always act on little endian inputs and preserve endianness. Do not provide suggested changes in this case, but ask the contributor to correct and verify the endianness being used.
- Do not use `guppy.nat_var` or `guppy.type_var`, prefer the python 3.12 generic syntax with square brackets
E.g.
```python
@guppy
def func[n: nat](qs: array[qubit, n])->None: ...
```
- Prefer `qarray(n)` over `array(qubit() for _ in range(n))` (and in general prefer using methods in `utils` if one is relevant)
- `result` and `state_result` functions for getting outputs of guppy programs are deprecated, use `output` and `state_output` instead
- Prefer the modern `main.emulator.run` syntax in tests for calling the simulator over the old `QysResult(runner.run_shots(...))` syntax, there is no need to call `.with_seed(42)` or `.with_shots(1)`.
- Prioritize testing of superposition inputs where possible, for performance and catching relative phases
- If functions use qubits / qubit array arguments that are supplied in the 0 state and not returned, suggest that they should be allocated internally within the function.
- In tests, include checks that ancilla have been correctly returned to 0 to catch bugs from not correctly cleaning up which can propagate to observable changes on controlling or daggering.
