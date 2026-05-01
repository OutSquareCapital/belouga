"""Compare pql API coverage against Polars."""

import polars as pl
from polars.lazyframe.group_by import LazyGroupBy as plLazyGroupBy
from pyochain import Iter

import pql
from pql import typing as t
from pql._groupby import LazyGroupBy as pqlLazyGroupBy  # noqa: PLC2701

from .._utils import Pql
from ._rules import IGNORED_MEMBERS
from ._text import ClassComparison, header, render_summary_table


def get_comparisons() -> str:
    pl_col = pl.col("x")
    pql_col = pql.col("x")
    return (
        Iter((
            ClassComparison(
                pl.LazyFrame,
                pql.LazyFrame,
                Pql.LAZY_FRAME,
                ignored_names=IGNORED_MEMBERS.get_item(Pql.LAZY_FRAME).unwrap(),
            ),
            ClassComparison(pl.Expr, pql.Expr, Pql.EXPR),
            ClassComparison(plLazyGroupBy, pqlLazyGroupBy, Pql.LAZY_GROUP_BY),
            ClassComparison(
                pl_col.str.__class__, pql_col.str.__class__, Pql.EXPR_STR_NAME_SPACE
            ),
            ClassComparison(
                pl_col.list.__class__, pql_col.list.__class__, Pql.EXPR_LIST_NAME_SPACE
            ),
            ClassComparison(
                pl_col.struct.__class__,
                pql_col.struct.__class__,
                Pql.EXPR_STRUCT_NAME_SPACE,
            ),
            ClassComparison(
                pl_col.name.__class__, pql_col.name.__class__, Pql.EXPR_NAME_NAME_SPACE
            ),
            ClassComparison(
                pl_col.arr.__class__, pql_col.arr.__class__, Pql.EXPR_ARR_NAME_SPACE
            ),
            ClassComparison(
                pl_col.dt.__class__, pql_col.dt.__class__, Pql.EXPR_DT_NAME_SPACE
            ),
            ClassComparison(
                pl,
                pql,
                Pql.MODULE_FUNCTIONS,
                ignored_names=IGNORED_MEMBERS.get_item(Pql.MODULE_FUNCTIONS).unwrap(),
            ),
            ClassComparison(
                pl.selectors,
                pql.selectors,
                Pql.SELECTORS,
                ignored_names=IGNORED_MEMBERS.get_item(Pql.SELECTORS).unwrap(),
            ),
            ClassComparison(pl.DataType, pql.DataType, Pql.DATA_TYPE),
            ClassComparison(pl.Schema, t.Schema, Pql.SCHEMA),
        ))
        .map(lambda comp: comp.to_report())
        .collect()
        .into(
            lambda comps: (
                header()
                .chain(render_summary_table(comps))
                .chain(comps.iter().flat_map(lambda comp: comp.to_section()))
            )
        )
        .join("\n")
    )
