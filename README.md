# guppyalgos

A library of reusable primitives for composing abstract quantum algorithms,
written in [guppy](https://github.com/Quantinuum/guppylang).

The `guppyalgos` package includes quantum arithmetic, state preparation, QROM,
block encoding, Hamiltonian simulation, and phase estimation. It's reusable
components let you choose circuit implementations when assembling an algorithm.

Any release before version 1.0.0 is experimental, and the API may change between releases.
Pin the package version for reproducible work and review the
[changelog](CHANGELOG.md) before upgrading.

## Getting started

Requires Python 3.12 or newer and [uv](https://docs.astral.sh/uv/).
Install from a source checkout:

```sh
git clone https://github.com/Quantinuum/guppyalgos.git
cd guppyalgos
uv sync
```

Explore the [example notebooks](examples/) and the
[getting-started guide](docs/getting-started.md) for usage and compilation examples.

See [documentation](https://docs.quantinuum.com/guppy/algorithms) for user guide.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, checks, and our
`gh stack` workflow for stacked pull requests.

## Citation

If you use this library in research, please cite it using [CITATION.cff](CITATION.cff)
and include the version or commit you used.

## License

The `guppyalgos` package is licensed under the [Apache License 2.0](LICENSE).
Bundled material with a separate license retains its own license terms.
