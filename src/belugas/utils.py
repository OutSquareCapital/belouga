"""Helpers for iterating over arguments that may or may not be iterables."""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum
from typing import TYPE_CHECKING, override

from pyochain import Iter, Option, Seq
from sqlglot import exp

if TYPE_CHECKING:
    from .typing import TryIter


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
            return Iter[T].once(val)
        case Iterable():
            return Iter(val)
        case _:
            return Iter[T].once(val)
