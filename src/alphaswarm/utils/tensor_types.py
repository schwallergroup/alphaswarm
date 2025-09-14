"""Tensor typing helper functions."""

from beartype import beartype
from environs import Env
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor

env = Env()
env.read_env()


def identity(f):
    """Identity function."""
    return f


class TorchTyping:
    """Tensor typing class."""

    def __init__(self, abstract_dtype):
        """Initialise the TorchTyping class."""
        self.abstract_dtype = abstract_dtype

    def __getitem__(self, shapes: str):
        """Return the abstract data type."""
        return self.abstract_dtype[Tensor, shapes]


Float = TorchTyping(Float)
Int = TorchTyping(Int)

should_typecheck = env.bool("TYPECHECK", False)

typecheck = jaxtyped(typechecker=beartype) if should_typecheck else identity
