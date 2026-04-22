from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import auto
from functools import partial
from typing import TYPE_CHECKING, NamedTuple, Self, TypedDict, Unpack

import pyochain as pc
from sqlglot import exp

from ._core import into_expr
from .utils import TryIter, UpperStrEnum, try_iter

if TYPE_CHECKING:
    from .typing import FrameMode, IntoExprColumn, WindowExclude


type FrameBound = int | Bounds | str


class Bounds(UpperStrEnum):
    PRECEDING = auto()
    FOLLOWING = auto()
    UNBOUNDED = auto()
    CURRENT = auto()
    ROW = auto()


class DirectionArgs(TypedDict):
    descending: TryIter[bool]
    nulls_last: TryIter[bool]


class FnArgs(TypedDict):
    fn_order_by: pc.Option[TryIter[IntoExprColumn]]
    fn_descending: TryIter[bool]
    fn_nulls_last: TryIter[bool]


class ClauseArgs(TypedDict):
    partition_by: pc.Option[list[exp.Expr]]
    order: pc.Option[exp.Order]
    spec: pc.Option[exp.WindowSpec]


class BoundArgs(TypedDict):
    frame_start: pc.Option[FrameBound]
    frame_end: pc.Option[FrameBound]
    exclude: pc.Option[WindowExclude]


def get_partition(
    partition_by: pc.Option[TryIter[IntoExprColumn]],
) -> pc.Option[list[exp.Expr]]:
    return partition_by.map(try_iter).map(
        lambda cols: cols.map(into_expr).collect(list)
    )


@dataclass(slots=True)
class OverBuilder:
    expr: exp.Expr

    def handle_distinct(self, *, distinct: bool) -> Self:
        match (distinct, self.expr.find(exp.Func)):
            case (True, exp.Func(this=arg)):  # pyright: ignore[reportAny]
                expr = self.expr.copy()
                expr.set("this", exp.Distinct(expressions=[arg]))
                return self.__class__(expr)
            case _:
                return self

    def handle_filter(self, filter_cond: pc.Option[IntoExprColumn]) -> Self:
        return (
            filter_cond
            .map(
                lambda c: exp.Filter(
                    this=self.expr, expression=exp.Where(this=into_expr(c))
                )
            )
            .map(self.__class__)
            .unwrap_or(self)
        )

    def handle_clauses(self, **kwargs: Unpack[ClauseArgs]) -> Self:
        expr, clauses = _rewrite_forward_fill(self.expr, kwargs)

        match expr.find(exp.Window):
            case None:
                return self.__class__(_wrap_in_window(expr, clauses))
            case _:
                return self.__class__(_inject_into_existing(expr, clauses))

    def build_fn(
        self, *, ignore_nulls: bool = False, **kwargs: Unpack[FnArgs]
    ) -> exp.Expr:
        return (
            self
            .handle_fn_order_by(**kwargs)
            .handle_nulls(ignore_nulls=ignore_nulls)
            .build()
        )

    def build(self) -> exp.Expr:
        return self.expr

    def handle_nulls(self, *, ignore_nulls: bool) -> Self:
        match ignore_nulls:
            case True:
                return self.__class__(exp.IgnoreNulls(this=self.expr))
            case False:
                return self

    def handle_fn_order_by(self, **kwargs: Unpack[FnArgs]) -> Self:
        def _build(cols: pc.Seq[IntoExprColumn]) -> exp.WithinGroup:
            exprs = _ordered(
                cols,
                descending=kwargs["fn_descending"],
                nulls_last=kwargs["fn_nulls_last"],
            )
            return exp.WithinGroup(
                this=self.expr, expression=exp.Order(expressions=exprs)
            )

        return (
            kwargs["fn_order_by"]
            .map(lambda x: try_iter(x).collect().into(_build))
            .map(self.__class__)
            .unwrap_or(self)
        )


def rolling_agg(expr: exp.Expr, order_by: str, spec: BoundsValues) -> exp.Expr:
    """Build a window expression with a prebuilt spec. Used by rolling aggregations.

    Returns:
        exp.Expr: The window expression with the rolling spec applied.
    """
    return (
        OverBuilder(expr)
        .handle_clauses(
            partition_by=pc.NONE,
            order=get_order(pc.Some(order_by), descending=False, nulls_last=False),
            spec=pc.Some(spec.into_spec("ROWS")),
        )
        .build()
    )


def _rewrite_forward_fill(
    expr: exp.Expr, clauses: ClauseArgs
) -> tuple[exp.Expr, ClauseArgs]:
    from ._meta import Marker

    def _last_value_arg(inner: exp.Expr) -> pc.Option[exp.Expr]:
        match inner:
            case exp.Anonymous() as fn if fn.name.lower() == "last_value":
                return pc.Some(fn.expressions[0])  # pyright: ignore[reportAny]
            case _:
                return pc.NONE

    def _rewritten(arg: exp.Expr) -> tuple[exp.AnyValue, ClauseArgs]:
        return (
            exp.AnyValue(this=arg),
            ClauseArgs(
                partition_by=clauses["partition_by"],
                order=get_order(
                    pc.Some(Marker.TEMP),
                    descending=True,
                    nulls_last=False,
                ),
                spec=make_spec(
                    "ROWS",
                    has_order_by=True,
                    frame_start=pc.Some(0),
                    frame_end=pc.NONE,
                    exclude=pc.NONE,
                ),
            ),
        )

    match expr:
        case exp.IgnoreNulls():
            match expr.args.get("this"):
                case exp.Expr() as val:
                    return _last_value_arg(val).map_or((expr, clauses), _rewritten)
                case _:
                    return expr, clauses
        case _:
            return expr, clauses


def get_order(
    order_by: pc.Option[TryIter[IntoExprColumn]], **kwargs: Unpack[DirectionArgs]
) -> pc.Option[exp.Order]:
    return order_by.map(lambda x: try_iter(x).collect()).map(
        lambda cols: exp.Order(expressions=_ordered(cols, **kwargs))
    )


def _ordered(
    cols: pc.Seq[IntoExprColumn], **kwargs: Unpack[DirectionArgs]
) -> list[exp.Ordered]:
    def _expand_clauses(*, clauses: TryIter[bool], n: int) -> pc.Iter[bool]:
        match clauses:
            case Iterable() as seq:
                return pc.Iter(seq)
            case _ as val:
                return try_iter(val).cycle().take(n)

    return (
        cols
        .iter()
        .zip(
            _expand_clauses(clauses=kwargs["descending"], n=cols.length()),
            _expand_clauses(clauses=kwargs["nulls_last"], n=cols.length()),
        )
        .map_star(
            lambda item, desc, nl: exp.Ordered(
                this=into_expr(item), desc=desc, nulls_first=not nl
            )
        )
        .collect(list)
    )


def _inject_into_existing(expr: exp.Expr, clauses: ClauseArgs) -> exp.Expr:
    inj = partial(_inject, clauses=clauses, with_spec=False)
    pc.Iter(expr.find_all(exp.Window)).for_each(inj)
    return expr


def _wrap_in_window(expr: exp.Expr, clauses: ClauseArgs) -> exp.Window:
    window = exp.Window(this=expr)
    _inject(window, clauses, with_spec=True)
    return window


def _inject(w: exp.Window, clauses: ClauseArgs, *, with_spec: bool) -> None:
    _ = clauses["partition_by"].map(lambda pb: w.set("partition_by", pb))
    _ = clauses["order"].map(lambda o: w.set("order", o))
    if with_spec:
        _ = clauses["spec"].map(lambda s: w.set("spec", s))


def make_spec(
    mode: FrameMode, *, has_order_by: bool, **bounds: Unpack[BoundArgs]
) -> pc.Option[exp.WindowSpec]:
    return (
        BoundsValues
        .new(bounds, has_order_by=has_order_by)
        .map(lambda b: b.into_spec(mode))
        .inspect(lambda spec: bounds["exclude"].map(lambda ex: spec.set("exclude", ex)))
    )


class Side(NamedTuple):
    value: FrameBound
    direction: FrameBound

    @classmethod
    def new(cls, value: FrameBound, direction: Bounds) -> Self:
        """Convert a frame bound value to ``(value, side)`` for `exp.WindowSpec`.

        Returns:
            Self: A Side instance with the value and direction for the window bound.
        """
        match value:
            case 0:
                return cls(Bounds.CURRENT, Bounds.ROW)
            case int(n):
                return cls(str(n), direction)
            case _:
                return cls(value, direction)


class BoundsValues(NamedTuple):
    start: Side
    end: Side

    @classmethod
    def rolling(cls, window_size: int, *, center: bool) -> Self:
        size = window_size - 1
        match center:
            case True:
                left = window_size // 2
                right = size - left
                return cls(
                    Side(str(left), Bounds.PRECEDING),
                    Side(str(right), Bounds.FOLLOWING),
                )
            case False:
                return cls(
                    Side(str(size), Bounds.PRECEDING), Side(Bounds.CURRENT, Bounds.ROW)
                )

    @classmethod
    def new(cls, bounds: BoundArgs, *, has_order_by: bool) -> pc.Option[Self]:
        match (bounds["frame_start"], bounds["frame_end"]):
            case (pc.Some(s), pc.Some(e)):
                return pc.Some(
                    cls(Side.new(s, Bounds.PRECEDING), Side.new(e, Bounds.FOLLOWING))
                )
            case (pc.Some(s), pc.NONE):
                return pc.Some(
                    cls(
                        Side.new(s, Bounds.PRECEDING),
                        Side.new(Bounds.UNBOUNDED, Bounds.FOLLOWING),
                    )
                )
            case (pc.NONE, pc.Some(e)):
                return pc.Some(
                    cls(
                        Side.new(Bounds.UNBOUNDED, Bounds.PRECEDING),
                        Side.new(e, Bounds.FOLLOWING),
                    )
                )
            case _ if has_order_by or bounds["exclude"].is_some():
                return pc.Some(
                    cls(
                        Side.new(Bounds.UNBOUNDED, Bounds.PRECEDING),
                        Side.new(Bounds.UNBOUNDED, Bounds.FOLLOWING),
                    )
                )
            case _:
                return pc.NONE

    def into_spec(self, mode: FrameMode) -> exp.WindowSpec:
        return exp.WindowSpec(
            kind=mode,
            start=self.start.value,
            start_side=self.start.direction,
            end=self.end.value,
            end_side=self.end.direction,
        )
