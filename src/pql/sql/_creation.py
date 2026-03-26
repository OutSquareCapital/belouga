from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from functools import partial
from typing import TYPE_CHECKING, Any, cast

import duckdb
import pyochain as pc
from sqlglot import exp

from ._funcs import unnest
from .typing import FrameLike, NPArrayLike, PythonLiteral

if TYPE_CHECKING:
    from narwhals.typing import IntoFrame

    from .typing import AnyArray, IntoDict, IntoRel, Orientation, SeqIntoVals

COL0 = "column_0"


def _named(j: object) -> str:
    return f"column_{j}"


def _to_expr(k: str, v: PythonLiteral) -> duckdb.Expression:
    return duckdb.ConstantExpression(v).alias(k)


def _unnest(k: str) -> duckdb.Expression:
    return unnest(k).alias(k).into_duckdb()


def into_relation(  # noqa: PLR0911
    data: IntoRel, orient: Orientation = "col"
) -> duckdb.DuckDBPyRelation:
    from .._frame import LazyFrame
    from ._core import DuckHandler

    match data:
        case duckdb.DuckDBPyRelation():
            return data
        case LazyFrame():
            return data.inner()
        case exp.Expr():
            return duckdb.values(DuckHandler(data).into_duckdb())
        case DuckHandler():
            return duckdb.values(data.into_duckdb())
        case Mapping():
            return from_dict(data)
        case NPArrayLike():
            return from_numpy(data, orient=orient)
        case FrameLike():
            return from_df(data)
        case Sequence():
            return from_records(data, orient=orient)


_QRY_ERR = "No relation provided"
_PY_CODE = partial(exec, "relation = dk.from_query(qry)")


def from_query(query: str, **relations: IntoRel) -> duckdb.DuckDBPyRelation:
    """Create a relation from a SQL query."""

    def _as_namespace(
        rels: IntoDict[str, duckdb.DuckDBPyRelation],
    ) -> duckdb.DuckDBPyRelation:
        namespace = {"dk": duckdb, "qry": query, **dict(rels)}
        _PY_CODE(locals=namespace)
        return cast(duckdb.DuckDBPyRelation, namespace["relation"])

    return (
        pc.Iter(relations.items())
        .map_star(lambda k, v: (k, into_relation(v)))
        .into(_as_namespace)
    )


def from_dicts(data: Sequence[Mapping[str, PythonLiteral]]) -> duckdb.DuckDBPyRelation:
    return (
        pc.Iter(data[0])
        .map(lambda key: (key, pc.Iter(data).map(lambda row: row[key]).collect(tuple)))
        .into(from_dict)
    )


def from_records(
    data: SeqIntoVals, orient: Orientation = "col"
) -> duckdb.DuckDBPyRelation:
    from ._core import DuckHandler

    match data[0]:
        case Mapping():
            vals = cast(Sequence[Mapping[str, Any]], data)  # pyright: ignore[reportExplicitAny]
            return from_dicts(vals)
        case Sequence() as value if not isinstance(value, str | bytes | bytearray):  # pyright: ignore[reportUnknownVariableType]
            vals = cast(Sequence[Sequence[PythonLiteral]], data)

            match orient:
                case "col":
                    return (
                        pc.Iter(vals)
                        .enumerate()
                        .map_star(lambda k, v: (_named(k), v))
                        .into(from_dict)
                    )

                case "row":
                    width = len(vals[0])
                    return (
                        pc.Iter(range(width))
                        .map(
                            lambda j: (
                                _named(j),
                                pc.Iter(vals).map(lambda row: row[j]).collect(tuple),
                            )
                        )
                        .into(from_dict)
                    )
        case exp.Expr():
            vals = cast(Sequence[exp.Expr], data)

            return duckdb.values(
                pc.Iter(vals).map(lambda e: DuckHandler(e).into_duckdb()).collect(tuple)
            )
        case _:
            return duckdb.values(_to_expr(COL0, tuple(data))).select(_unnest(COL0))


def from_df(data: IntoFrame) -> duckdb.DuckDBPyRelation:
    import narwhals as nw

    match nw.from_native(data):
        case nw.DataFrame() as df:
            return df.lazy(backend="duckdb").to_native()  # pyright: ignore[reportAny]
        case nw.LazyFrame() as lf:
            return duckdb.from_arrow(lf.collect())


def from_dict(data: IntoDict[str, Any]) -> duckdb.DuckDBPyRelation:  # pyright: ignore[reportExplicitAny]
    data = pc.Dict(data)

    raw_vals = data.items().iter().map_star(_to_expr).collect(tuple)
    return duckdb.values(raw_vals).select(*data.iter().map(_unnest))


def from_numpy(data: AnyArray, orient: Orientation = "col") -> duckdb.DuckDBPyRelation:

    match data.ndim:
        case 1:
            return duckdb.values(_to_expr(COL0, data)).select(_unnest(COL0))
        case _:
            arr = data.T if orient == "col" else data

            def _array_strategy() -> tuple[int, Callable[[int], AnyArray]]:
                match (arr.ndim, orient):
                    case (2, _) | (_, "row"):
                        return 1, lambda j: arr[:, j]  # pyright: ignore[reportAny]
                    case _:
                        return 0, lambda j: arr[j]  # pyright: ignore[reportAny]

            def _named_array(names: pc.Seq[str]) -> duckdb.DuckDBPyRelation:
                vals = (
                    names.iter()
                    .enumerate()
                    .map_star(lambda j, name: _to_expr(name, _arr_getter(j)))
                    .collect(tuple)
                )

                return duckdb.values(vals).select(*names.iter().map(_unnest))

            axis, _arr_getter = _array_strategy()
            names_nb: int = arr.shape[axis]  # pyright: ignore[reportAny]
            return pc.Iter(range(names_nb)).map(_named).collect().into(_named_array)
