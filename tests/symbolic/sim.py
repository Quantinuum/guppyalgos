"""A symbolic interpreter for the HUGR of classical reversible circuits in guppy."""

from hugr import ops, val
from hugr.std.collections.static_array import StaticArrayVal

from guppyalgos.primitives.gate_decompositions.and_op import temp_and
from guppyalgos.primitives.gate_decompositions.cnx.cnx import _rtof4
from tests.symbolic.symint import xor


def and_compute(sim, a, b, t):
    """Run the AND gadget as a Toffoli onto a target that must be |0>."""
    if t != t.bdd.false:
        raise AssertionError("the AND gadget needs its target in |0>")
    return [a, b, a & b]


def and_uncompute(sim, a, b, t):
    """Measure out and reset the target of the AND gadget, which must hold the AND."""
    if t != a & b:
        raise AssertionError("the AND gadget is uncomputed on a wrong target")
    return [a, b, t.bdd.false]


def rtof4_compute(sim, a, b, c, t):
    """Run a relative-phase CCCX of controls a, b and c onto a target in |0>."""
    if t != t.bdd.false:
        raise AssertionError("the relative-phase CCCX needs its target in |0>")
    sim.pending.append((a & b, c))
    return [a, b, c, a & b & c]


def rtof4_uncompute(sim, a, b, c, t):
    """Undo a relative-phase CCCX of controls a, b and c, which removes its phase."""
    if t != a & b & c or (a & b, c) not in sim.pending:
        raise AssertionError(
            "the relative-phase CCCX is undone on other controls or a wrong target"
        )
    sim.pending.remove((a & b, c))
    return [a, b, c, t.bdd.false]


def discard(q):
    """Discard a qubit, which must be back in |0>."""
    if q != q.bdd.false:
        raise AssertionError("a qubit is discarded while not in |0>")


def reset(q):
    """Reset a qubit to |0>, which may hold |1> but must not depend on the inputs."""
    if q.bdd.support(q):
        raise AssertionError("a qubit is reset while it depends on the inputs")
    return [q.bdd.false]


def measure(q):
    """Measure a qubit, which must hold a fixed value."""
    if q not in (q.bdd.true, q.bdd.false):
        raise AssertionError("a qubit is measured while its value is not fixed")
    return [boolean(q == q.bdd.true)]


def boolean(b):
    """Return a HUGR boolean."""
    return (int(b), [])


def put(arr, i, x):
    """Return a copy of ``arr`` with ``x`` at index ``i``."""
    return [*arr[:i], x, *arr[i + 1 :]]


def hugr_name(definition):
    """Return the name that a guppy function has in the HUGR it compiles to."""
    f = definition.wrapped.python_func
    return f"{f.__module__}.{f.__qualname__}"


GADGETS = {  # a unitary class compiles to its methods, such as __call__ and daggered
    hugr_name(temp_and): and_compute,
    hugr_name(temp_and).replace("__call__", "daggered"): and_uncompute,
    hugr_name(_rtof4): rtof4_compute,
    hugr_name(_rtof4).replace("__call__", "daggered"): rtof4_uncompute,
}

COUNT = {  # how each op changes the number of qubits allocated and not freed
    "tket.quantum.QAlloc": 1,
    "tket.quantum.QFree": -1,
    "tket.quantum.MeasureFree": -1,
}

OPS = {
    "tket.quantum.X": lambda q: [~q],
    "tket.quantum.CX": lambda c, t: [c, xor(t, c)],
    "tket.quantum.Toffoli": lambda a, b, t: [a, b, xor(t, a & b)],
    "tket.quantum.QFree": discard,
    "tket.quantum.Reset": reset,
    "tket.quantum.MeasureFree": measure,
    "tket.measurement.Read": lambda b: [b],
    "tket.guppy.drop": lambda x: None,
    "arithmetic.int.iadd": lambda a, b: [a + b],
    "arithmetic.int.isub": lambda a, b: [a - b],
    "arithmetic.int.imul": lambda a, b: [a * b],
    "arithmetic.int.idiv_s": lambda a, b: [a // b],
    "arithmetic.int.imod_s": lambda a, b: [a % b],
    "arithmetic.int.ipow": lambda a, b: [a**b],
    "arithmetic.int.ieq": lambda a, b: [boolean(a == b)],
    "arithmetic.int.ine": lambda a, b: [boolean(a != b)],
    "arithmetic.int.ilt_s": lambda a, b: [boolean(a < b)],
    "arithmetic.int.ile_s": lambda a, b: [boolean(a <= b)],
    "arithmetic.int.igt_s": lambda a, b: [boolean(a > b)],
    "arithmetic.int.ige_s": lambda a, b: [boolean(a >= b)],
    "arithmetic.int.iu_to_s": lambda a: [a],
    "arithmetic.conversions.itousize": lambda a: [a % 2**64],
    "collections.borrow_arr.new_array": lambda *xs: [list(xs)],
    "collections.borrow_arr.discard_all_borrowed": lambda arr: None,
    "collections.borrow_arr.borrow": lambda arr, i: [put(arr, i, None), arr[i]],
    "collections.borrow_arr.return": lambda arr, i, x: [put(arr, i, x)],
    "collections.borrow_arr.is_borrowed": lambda arr, i: [arr, boolean(arr[i] is None)],
    "collections.borrow_arr.get": lambda arr, i: [(1, [arr[i]]), arr],
    "collections.borrow_arr.set": lambda arr, i, x: [(1, [arr[i], put(arr, i, x)])],
    "collections.borrow_arr.unpack": lambda arr: arr,
    "collections.borrow_arr.clone": lambda arr: [arr, list(arr)],
    "collections.borrow_arr.to_array": lambda arr: [arr],
    "collections.borrow_arr.from_array": lambda arr: [arr],
    "collections.array.new_array": lambda *xs: [list(xs)],
    "collections.array.unpack": lambda arr: arr,
    "collections.static_array.get": lambda arr, i: [(1, [arr[i]])],
}


def const(v):
    """Decode a HUGR constant."""
    if isinstance(v, val.Sum):
        return (v.tag, [const(x) for x in v.vals])
    if isinstance(v, StaticArrayVal):
        return [const(x) for x in v.v]
    if isinstance(v, val.Extension) and v.name == "StaticArrayValue":
        return [unserialize(x) for x in v.val["value"]["values"]]
    return v.v if isinstance(getattr(v, "v", None), int) else v


def unserialize(d):
    """Decode an element of a static array that the modifier pass left serialized."""
    if d["v"] in ("Sum", "Tuple"):
        return (d.get("tag", 0), [unserialize(x) for x in d["vs"]])
    if d["value"]["c"] == "StaticArrayValue":
        return [unserialize(x) for x in d["value"]["v"]["value"]["values"]]
    # an integer is stored as its unsigned bit pattern, as hugr's IntVal does
    payload = d["value"]["v"]
    width, value = 1 << payload["log_width"], int(payload["value"])
    return value - (1 << width) if value >> width - 1 else value


class Sim:
    """Walk a HUGR, running classical ops concretely and gates on boolean functions."""

    def __init__(self, hugr, bdd):
        """Prepare the walk of ``hugr`` on boolean functions of ``bdd``."""
        self.hugr, self.bdd = hugr, bdd
        self.nodes = {}  # the nodes of each region in order, with their op and inputs
        self.pending = []  # (a & b, c) of each relative-phase CCCX not undone yet
        self.ancillas = self.peak = 0  # the qubits allocated now, and at most

    def source(self, port):
        """Return the key of a port in ``env``, and its value if no node sets it."""
        op = self.hugr[port.node].op
        outside = const(op.val) if isinstance(op, ops.Const) else port.node
        return (port.node.idx, port.offset), outside

    def call(self, f, args):
        """Run a function, or its model if it is one of the gadgets."""
        name = self.hugr[f].op.f_name
        model = GADGETS.get(name)
        try:
            return model(self, *args) if model else self.run(f, args)
        except Exception as error:
            error.add_note(f"in {name}")
            raise

    def run(self, region, ins):
        """Evaluate a dataflow region and return what reaches its Output node."""
        env = {}
        if region not in self.nodes:
            self.nodes[region] = []
            for node in self.hugr.sorted_region_nodes(region):
                links = self.hugr.incoming_links(node)
                ports = [s[0] for p, s in links if p.offset >= 0]
                sources = [self.source(p) for p in ports]
                self.nodes[region].append((node, self.hugr[node].op, sources))
        for node, op, sources in self.nodes[region]:
            args = [env.get(key, outside) for key, outside in sources]
            match type(op):
                case ops.Input:
                    outs = ins
                case ops.Output:
                    result, outs = args, []
                case ops.Const:
                    continue
                case ops.LoadConst | ops.LoadFunc:
                    outs = args
                case ops.Call:
                    outs = self.call(args[-1], args[:-1])
                case ops.CallIndirect:
                    outs = self.call(args[0], args[1:])
                case ops.MakeTuple:
                    outs = [(0, args)]
                case ops.UnpackTuple:
                    outs = args[0][1]
                case ops.Tag:
                    outs = [(op.tag, args)]
                case ops.Conditional:
                    (tag, fields), *rest = args
                    outs = self.run(self.hugr.children(node)[tag], fields + rest)
                case ops.TailLoop:
                    tag, outs = 0, args
                    while not tag:
                        (tag, fields), *rest = self.run(node, outs)
                        outs = fields + rest
                case ops.CFG:
                    block, exit_block = self.hugr.children(node)[:2]
                    outs = args
                    while block != exit_block:
                        (tag, fields), *rest = self.run(block, outs)
                        block = next(iter(self.hugr.linked_ports(block.out(tag)))).node
                        outs = fields + rest
                case ops.ExtOp:
                    name = op.op_def().qualified_name()
                    kind = name.removeprefix("collections.borrow_arr.")
                    if kind == "discard_all_borrowed":
                        assert all(x is None for x in args[0]), "an array is not empty"
                    elif kind in ("borrow", "get", "set", "return"):
                        arr, i = args[:2]
                        assert i < len(arr), "an array index is out of range"
                        if kind == "return":
                            assert arr[i] is None, "an array slot is full"
                        else:
                            assert arr[i] is not None, "an array slot is empty"
                    self.ancillas += COUNT.get(name, 0)
                    self.peak = max(self.peak, self.ancillas)
                    if name == "tket.quantum.QAlloc":
                        outs = [self.bdd.false]
                    elif name == "collections.borrow_arr.new_all_borrowed":
                        outs = [[None] * op.args[0].n]
                    elif name in OPS:
                        outs = OPS[name](*args) or []
                    else:
                        raise NotImplementedError(f"op {name}")
                case _:
                    raise NotImplementedError(f"HUGR op {type(op).__name__}")
            for i, v in enumerate(outs):
                env[node.idx, i] = v
        return result
