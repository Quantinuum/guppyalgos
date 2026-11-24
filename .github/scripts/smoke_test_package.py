"""Check an installed wheel and compile a small program outside the source tree."""

import pkgutil
import sys
from importlib import import_module
from pathlib import Path

from guppylang import guppy
from guppylang.std.quantum import discard_array

import guppyalgos
from guppyalgos.primitives.state_preparation import uniform_state
from guppyalgos.utils import qarray

assert Path(guppyalgos.__file__).is_relative_to(Path(sys.prefix)), (
    "Expected guppyalgos to be imported from the isolated environment"
)
modules = list(pkgutil.walk_packages(guppyalgos.__path__, guppyalgos.__name__ + "."))
for module in modules:
    import_module(module.name)

uniform = uniform_state(4)


@guppy
def main() -> None:
    """Compile a two-qubit uniform state preparation circuit."""
    register = qarray(2)
    uniform(register)
    discard_array(register)


main.compile()
print(f"Imported {len(modules)} modules and compiled a Guppy program.")
