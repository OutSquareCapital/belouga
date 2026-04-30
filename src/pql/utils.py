"""Helpers for iterating over arguments that may or may not be iterables."""

from collections.abc import Iterable, Sequence
from enum import StrEnum
from typing import Any, override

from pyochain import Err, Iter, Ok, Option, Result, Seq
from sqlglot import exp

from .typing import NonNestedLiteral

type TryIter[T] = Iterable[T] | T | None
"""Represent a value that may or may not be an `Iterable`."""
type TrySeq[T] = Sequence[T] | T | None
"""Represent a value that may or may not be a `Sequence`."""


class UpperStrEnum(StrEnum):
    """A `StrEnum` that automatically converts its values to uppercase."""

    @override
    @staticmethod
    def _generate_next_value_(
        name: str, start: object, count: object, last_values: object
    ) -> str:
        return name.upper()


def try_seq[T](val: TryIter[T]) -> Option[Seq[T]]:
    """Try to convert a potentially iterable value to an `Option[Seq]`.

    Args:
        val (TryIter[T]): The value to try to convert.

    Returns:
        Option[Seq[T]]: `Some(Seq)` if the value is iterable, otherwise `None`.
    """
    return try_iter(val).collect().then_some()


def check_by_arg[T: NonNestedLiteral](
    compared: Seq[Any],  # pyright: ignore[reportExplicitAny]
    name: str,
    arg: TrySeq[T],
) -> Result[Iter[T], ValueError]:
    """Checks if the sequence arg matches the length of compared.

    Returns an iterator over arg if lengths match, otherwise returns a ValueError.

    If arg is not a sequence, repeats its value to match the length of compared.

    Returns:
        Result[Iter[T], ValueError]: An iterator over the values in arg if the length of arg matches the length of compared, otherwise a ValueError.
    """
    length = compared.length()
    match arg:
        case Sequence():
            len_arg = len(arg)
            match len_arg == length:
                case True:
                    return Ok(try_iter(arg))
                case False:
                    msg = f"the length of `{name}` ({len_arg}) does not match the length of `by` ({length})"
                    return Err(ValueError(msg))

        case _:
            return Ok(try_iter(arg).cycle().take(length))


def try_iter[T](val: TryIter[T]) -> Iter[T]:
    """Try to iterate over a value that may or may not be iterable.

    Args:
        val (TryIter[T]): The value to try to iterate over.

    Returns:
        Iter[T]: An iterator over the value if it is iterable, otherwise an iterator over a single element.
    """
    match val:
        case None:
            return Iter[T].new()
        case str() | bytes() | bytearray() | exp.Expr():
            return Iter[T].once(val)  # pyright: ignore[reportReturnType]
        case Iterable():
            return Iter(val)  # pyright: ignore[reportUnknownArgumentType]
        case _:
            return Iter[T].once(val)
