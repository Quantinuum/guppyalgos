"""A formal verifier for reversible circuits in guppy, by symbolic execution on BDDs."""

import inspect
from collections.abc import Callable

from dd.autoref import BDD
from guppylang_internals.engine import ENGINE
from hugr import tys
from hugr.metadata import HugrUsedExtensions
from tket.passes import ModifierResolverPass

from tests.symbolic.sim import Sim, hugr_name
from tests.symbolic.symint import SymInt, xor


def compile_together(fns):
    """Compile guppy functions into one HUGR module, faster than one by one."""
    hugr = ENGINE.compile([fn.id for fn in fns]).package.modules[0]
    extensions = hugr[hugr.module_root].metadata[HugrUsedExtensions]
    if any(ext.name == "tket.modifier" for ext in extensions):
        ModifierResolverPass().run(hugr)
    return hugr


def size(ty):
    """Return the number of qubits in an argument of type ``ty``."""
    return 1 if ty == tys.Qubit else ty.args[0].n * size(ty.args[1].ty)


def argument(ty, qubits, register=list):
    """Put qubits, in order, into ``ty``, using ``register`` for arrays of qubits."""
    if ty == tys.Qubit:
        return next(qubits)
    elem = ty.args[1].ty
    items = [argument(elem, qubits, register) for _ in range(ty.args[0].n)]
    return register(items) if elem == tys.Qubit else items


def bits(x):
    """Return the boolean function of each qubit of an argument, in order."""
    return [y for z in x for y in bits(z)] if isinstance(x, list | SymInt) else [x]


def verify(
    fn,
    spec: Callable[..., tuple],
    zero: tuple[str, ...] = (),
    hugr=None,
    ancillas: int | None = None,
) -> None:
    """Check a guppy function against a specification on every input.

    Args:
        fn: The guppy function to check.
        spec: A Python function with one argument per argument of ``fn``,
            returning the expected output of each as a tuple. Each array of
            qubits comes as a ``SymInt``, and each qubit as a boolean function.
        zero: A tuple of the names of the arguments that start in |0> instead
            of being symbolic, such as a clean ancilla.
        hugr: A module from ``compile_together`` that holds ``fn``. By default,
            ``fn`` is compiled alone.
        ancillas: The most qubits ``fn`` holds at once besides its arguments. By
            default, it is not checked.

    Raises:
        AssertionError: If an output differs from the specification, if the
            circuit is not reversible, such as when an ancilla is discarded dirty,
            or if the ancilla count differs from ``ancillas``.

    """
    # compile the function, find it in its module and read its signature
    if hugr is None:
        hugr = compile_together([fn])
    f_name = hugr_name(fn)
    nodes = hugr.children(hugr.module_root)
    (node,) = (n for n in nodes if getattr(hugr[n].op, "f_name", None) == f_name)
    signature = inspect.signature(fn.wrapped.python_func)
    names = list(signature.parameters)
    assert signature.return_annotation is None, "fn must return nothing"
    assert set(zero) <= set(names), f"no argument is named {set(zero) - set(names)}"
    types = hugr[node].op.inputs

    # create the BDD variables, interleaving the bits of the arguments
    bdd = BDD()
    regs = [
        [None if arg in zero else f"{arg}[{k}]" for k in range(size(ty))]
        for arg, ty in zip(names, types, strict=True)
    ]
    for k in range(max(map(len, regs), default=0)):
        bdd.declare(*(reg[k] for reg in regs if k < len(reg) and reg[k]))

    # build the symbolic inputs
    qubits = [[bdd.var(v) if v else bdd.false for v in reg] for reg in regs]
    args = [argument(ty, iter(qs)) for ty, qs in zip(types, qubits, strict=True)]
    inputs = [
        argument(ty, iter(qs), SymInt) for ty, qs in zip(types, qubits, strict=True)
    ]

    # run the circuit
    sim = Sim(hugr, bdd)
    outputs = sim.run(node, args)
    if sim.pending:
        raise AssertionError("a relative-phase CCCX is never uncomputed")
    if ancillas is not None and sim.peak != ancillas:
        raise AssertionError(f"the ancilla count is {sim.peak}, not {ancillas}")

    # compare every output bit with the specification
    for name, got, want in zip(names, outputs, spec(*inputs), strict=True):
        for k, (x, y) in enumerate(zip(bits(got), bits(want), strict=True)):
            if x != y:
                env = bdd.pick(xor(x, y), care_vars=bdd.vars) or {}
                vals = (sum(env[v] << j for j, v in enumerate(r) if v) for r in regs)
                counterexample = ", ".join(
                    f"{n}={v}" for n, v in zip(names, vals, strict=True)
                )
                raise AssertionError(
                    f"{name} bit {k} is wrong, counterexample: {counterexample}"
                )
