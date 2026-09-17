"""Function utils."""

from guppylang.defs import GuppyFunctionDefinition
from hugr.model import Apply, DefineFunc, List as HugrList, Literal


def register_size(function: GuppyFunctionDefinition, parameter_index: int) -> int:
    """Determine the size of a fixed-size array parameter in a Guppy function.

    Args:
        function: The Guppy function definition to inspect.
        parameter_index: The index of the array parameter whose size is to be
            determined.

    Returns:
        The size of the specified fixed-size array parameter.

    Raises:
        TypeError: If the specified parameter is not a fixed-size array or if the array
            size is not concrete.

    """
    package = function.compile_function().to_model()
    function_node = package.modules[0].root.children[0]
    operation = function_node.operation
    if not isinstance(operation, DefineFunc):
        raise TypeError("Expected a compiled Guppy function definition.")
    signature = operation.symbol.signature
    if not isinstance(signature, Apply) or not isinstance(signature.args[0], HugrList):
        raise TypeError("Expected a concrete Guppy function signature.")
    parameter_type = signature.args[0].parts[parameter_index]
    if (
        not isinstance(parameter_type, Apply)
        or "borrow_array" not in parameter_type.symbol
    ):
        raise TypeError(
            f"Expected parameter {parameter_index + 1} to be a fixed-size array."
        )
    array_size = parameter_type.args[0]
    if not isinstance(array_size, Literal) or not isinstance(array_size.value, int):
        raise TypeError("Expected a concrete array size.")
    return array_size.value
