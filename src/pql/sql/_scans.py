from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from operator import itemgetter as get
from typing import TYPE_CHECKING, Any, Self, cast

import duckdb
import pyochain as pc

from ._funcs import unnest
from .typing import FrameLike, LitSeq, NestedSeq, NPArrayLike, PythonLiteral

if TYPE_CHECKING:
    from narwhals.typing import IntoFrame
    from sqlglot import exp

    from .._frame import LazyFrame
    from .typing import AnyArray, IntoDict, IntoRel, Orientation, SeqIntoVals

COL0 = "column_0"


def _single_col(name: str = COL0) -> pc.Vec[str]:
    return pc.Vec.from_ref([name])


def _named(j: object) -> str:
    return f"column_{j}"


def _into_tup[T](
    vals: Iterable[Sequence[T]] | Iterable[Mapping[T, object]], key: T
) -> tuple[T, ...]:
    return pc.Iter(vals).map(get(key)).collect(tuple)


def _to_expr(k: str, v: PythonLiteral) -> duckdb.Expression:
    return duckdb.ConstantExpression(v).alias(k)


def _unnest(k: str) -> duckdb.Expression:
    return duckdb.SQLExpression(unnest(k).alias(k).inner.sql(dialect="duckdb"))


class PQLConversionError(ValueError):
    """Raised when a conversion from a sqlglot expression to a DuckDB expression fails."""

    def __init__(self, e: Exception, expr: exp.Expr) -> None:
        msg = f"""
Failed to convert expression to DuckDB!
error:
        {e}
    expression:
        {expr!r}
    SQL:
        {expr.sql(dialect="duckdb", pretty=True, identify=True)}
"""
        super().__init__(msg)


@dataclass(slots=True)
class ScanSource:
    relation: duckdb.DuckDBPyRelation
    columns: pc.Vec[str]

    @classmethod
    def from_query(cls, query: exp.Expr, **relations: IntoRel) -> Self:
        """Create a `duckdb.DuckDBPyRelation` from a  `exp.Expr` node.

        Args:
            query (exp.Expr): The SQL node to execute.
            **relations (IntoRel): Relations to include in the query.

        Returns:
            duckdb.DuckDBPyRelation: The resulting DuckDB relation.

        Raises:
                PQLConversionError: If the SQL query cannot be parsed by DuckDB.
        """
        try:
            rels = (
                pc
                .Iter(relations.items())
                .map_star(lambda k, v: (k, cls.build(v).relation))
                .collect(dict)
            )
            parsed = query.sql(dialect="duckdb", identify=True)
            namespace = {"duckdb": duckdb, "parsed": parsed, **rels}
            exec("relation = duckdb.from_query(parsed)", namespace)
            result = cast(duckdb.DuckDBPyRelation, namespace["relation"])
            return cls.from_relation(result)
        except duckdb.ParserException as e:
            raise PQLConversionError(e, query) from e

    @classmethod
    def build(cls, source: IntoRel | None, orient: Orientation = "col") -> Self:  # noqa: PLR0911
        from .._frame import LazyFrame

        match source:
            case None:
                return cls.from_none()
            case ScanSource():
                return cls.copy(source)  # pyright: ignore[reportArgumentType]
            case duckdb.DuckDBPyRelation():
                return cls.from_relation(source)
            case LazyFrame():
                return cls.from_lf(source)
            case Mapping():
                return cls.from_dict(source)
            case NPArrayLike():
                return cls.from_numpy(source, orient=orient)
            case FrameLike():
                return cls.from_df(source)
            case Sequence():
                return cls.from_records(source, orient=orient)

    def into_frame(self) -> LazyFrame:
        from .._frame import LazyFrame

        return LazyFrame(self.relation)

    def copy(self) -> Self:
        return self.__class__(self.relation, self.columns.into(pc.Vec))

    @classmethod
    def from_lf(cls, lf: LazyFrame) -> Self:
        return cls(lf.inner.relation, lf.columns)

    @classmethod
    def from_records(cls, data: SeqIntoVals, orient: Orientation = "col") -> Self:
        match data[0]:
            case Mapping():
                vals = cast(Sequence[Mapping[str, Any]], data)  # pyright: ignore[reportExplicitAny]
                return cls.from_dicts(vals)
            case Sequence() as value if not isinstance(value, str | bytes | bytearray):  # pyright: ignore[reportUnknownVariableType]
                vals = cast(NestedSeq, data)
                match orient:
                    case "col":
                        return cls.from_seq_col(vals)

                    case "row":
                        return cls.from_seq_row(vals)
            case _:
                vals = cast(LitSeq, data)
                return cls.from_seq_lit(vals)

    @classmethod
    def from_dicts(cls, data: Sequence[Mapping[str, PythonLiteral]]) -> Self:
        return (
            pc
            .Iter(data[0])
            .map(lambda key: (key, _into_tup(data, key)))
            .into(cls.from_dict)
        )

    @classmethod
    def from_seq_lit(cls, data: LitSeq) -> Self:
        rel = duckdb.values(_to_expr(COL0, tuple(data))).select(_unnest(COL0))
        return cls(rel, _single_col())

    @classmethod
    def from_seq_col(cls, data: NestedSeq) -> Self:
        return (
            pc
            .Iter(data)
            .enumerate()
            .map_star(lambda k, v: (_named(k), v))
            .into(cls.from_dict)
        )

    @classmethod
    def from_seq_row(cls, data: NestedSeq) -> Self:
        width = len(data[0])
        return (
            pc
            .Iter(range(width))
            .map(lambda j: (_named(j), _into_tup(data, j)))
            .into(cls.from_dict)
        )

    @classmethod
    def from_df(cls, data: IntoFrame) -> Self:
        import narwhals as nw

        match nw.from_native(data):
            case nw.DataFrame() as df:
                rel = df.lazy(backend="duckdb").to_native()  # pyright: ignore[reportAny]
            case nw.LazyFrame() as lf:
                rel = duckdb.from_arrow(lf.collect())
        return cls.from_relation(rel)

    @classmethod
    def from_numpy(cls, data: AnyArray, orient: Orientation = "col") -> Self:

        match data.ndim:
            case 1:
                rel = duckdb.values(_to_expr(COL0, data)).select(_unnest(COL0))
                return cls(rel, _single_col())
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
                        names
                        .iter()
                        .enumerate()
                        .map_star(lambda j, name: _to_expr(name, arr_getter(j)))
                        .collect(tuple)
                    )

                    return duckdb.values(vals).select(*names.iter().map(_unnest))

                axis, arr_getter = _array_strategy()
                names_nb: int = arr.shape[axis]  # pyright: ignore[reportAny]
                cols = pc.Iter(range(names_nb)).map(_named).collect(pc.Vec)
                return cls(cols.into(_named_array), cols)

    @classmethod
    def from_table(cls, name: str) -> Self:
        return cls.from_relation(duckdb.table(name))

    @classmethod
    def from_table_function(cls, name: str, *args: object) -> Self:
        return cls.from_relation(duckdb.table_function(name, *args))

    @classmethod
    def from_relation(cls, relation: duckdb.DuckDBPyRelation) -> Self:
        return cls(relation, pc.Vec.from_ref(relation.columns))

    @classmethod
    def from_none(cls) -> Self:
        from ._meta import Marker

        return cls.from_dict({Marker.TEMP: ()})

    @classmethod
    def from_dict(cls, data: IntoDict[str, Any]) -> Self:  # pyright: ignore[reportExplicitAny]
        data = pc.Dict(data)

        raw_vals = data.items().iter().map_star(_to_expr).collect(tuple)
        rel = duckdb.values(raw_vals).select(*data.iter().map(_unnest))
        return cls(rel, data.keys().into(pc.Vec))
