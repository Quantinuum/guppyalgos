# guppyalgos documentation

[guppyalgos](https://github.com/Quantinuum/guppyalgos) is a
library of reusable quantum-algorithm components built on
[guppy](https://docs.quantinuum.com/guppy/language_guide/language_guide_index.html). It combines
type-safe guppy primitives with Python builders for algorithms that start from
classical data.

Any release before version 1.0.0 is experimental, and the API may change between releases.
Pin the package version for reproducible work and review the
[changelog](https://github.com/Quantinuum/guppyalgos/blob/main/CHANGELOG.md) before upgrading.

Start with **Getting started** to compile a first program. The user guide
introduces the library and its composition patterns; the notebooks provide
runnable workflows, while the API reference documents individual components.

**See the library in action:** {doc}`examples/block_encoding/block_encoding_demo`
starts with a two-term LCU, builds a quantum walk, and applies QSVT. It then
uses an eight-term Hamiltonian and introduces alias-sampling preparation
with configurable QROM fanout.

**Explore phase estimation:** {doc}`examples/phase_estimation/phase_estimation_demo`
starts with one qubit, then estimates the same Hamiltonian's energy using
Trotterized evolution and qubitization. It includes measured results and a
route to the advanced THC example.


**Explore the building blocks:** {doc}`arithmetic` covers reversible register
operations, and {doc}`measurement` covers Pauli measurements and expectation
estimation.

```{toctree}
:maxdepth: 2
:titlesonly:

getting-started.md
user-guide.md
```

```{toctree}
:maxdepth: 1
:titlesonly:

api/api.md
examples_index.md
```
