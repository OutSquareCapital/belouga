from collections.abc import Iterable

import sqlglot.expressions as exp
from pyochain import Iter, Option

from .typing import IntoExpr


def args_into_glot(args: Iterable[IntoExpr], *, as_col: bool = False) -> list[exp.Expr]:
    """Convert an `Iterable` of `IntoExpr` values into a list of sqlglot `Expr` nodes.

    Args:
        args (Iterable[IntoExpr]): The values to convert.
        as_col (bool): Whether to treat string values as column names. Defaults to `False`.

    Returns:
        list[exp.Expr]: A list of sqlglot expressions.
    """
    return (
        Iter(args)
        .filter_map(Option)
        .map(lambda x: into_glot(x, as_col=as_col))
        .collect(list)
    )


def into_glot(value: IntoExpr, *, as_col: bool = True) -> exp.Expr:
    """Convert an `IntoExpr` value into a sqlglot `Expr` node.

    Args:
        value (IntoExpr): The value to convert.
        as_col (bool): Whether to treat string values as column names. Defaults to `True`.

    Returns:
        exp.Expr: The resulting sqlglot expression.
    """
    from ._core import DuckHandler

    match value:
        case DuckHandler():
            return value.inner
        case str() if as_col:
            return exp.column(value)
        case _:
            return exp.convert(value)
