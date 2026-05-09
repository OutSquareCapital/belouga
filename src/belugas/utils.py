"""Helpers for iterating over arguments that may or may not be iterables."""

from collections.abc import Iterable, Sequence
from enum import StrEnum
from typing import override

from pyochain import Iter, Option, Seq
from sqlglot import exp

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


def select(exprs: Iterable[exp.Expr | str]) -> exp.Select:
    """Create a `SELECT` expression.

    Returns:
        exp.Select
    """
    return exp.select(*exprs)
