from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from pyochain import NONE, Dict, Iter, Null, Option, Some

from ..utils import try_iter
from . import nodes

if TYPE_CHECKING:
    from ..typing import IntoExpr, IntoExprColumn

type NewNode = Option[nodes.Node]


def optimize_nodes(plan_node: nodes.Node) -> nodes.Node:
    optimized_children = _optimize_children(plan_node)
    match optimized_children:
        case nodes.LogicalNode() as logical:
            match _flatten_pair(logical.inner, logical):
                case Some(merged):
                    return optimize_nodes(merged)
                case Null():
                    return optimized_children
        case _:
            return optimized_children


def _optimize_children(node: nodes.Node) -> nodes.Node:
    match node:
        case nodes.BaseScan():
            return node
        case nodes.Union() | nodes.Join() | nodes.JoinCross() | nodes.JoinAsof():
            return replace(
                node,
                inner=optimize_nodes(node.inner),
                other=optimize_nodes(node.other),
            )
        case nodes.LogicalNode():
            return replace(node, inner=optimize_nodes(node.inner))


def _flatten_pair(prev: nodes.Node, nxt: nodes.Node) -> NewNode:
    match prev, nxt:
        case nodes.Filter() as lhs, nodes.Filter() as rhs:
            return Some(_merge_filters(lhs, rhs))
        case nodes.Drop() as lhs, nodes.Drop() as rhs:
            return Some(_merge_drops(lhs, rhs))
        case nodes.Rename() as lhs, nodes.Rename() as rhs:
            return Some(_merge_renames(lhs, rhs))
        case nodes.Limit() as lhs, nodes.Limit() as rhs:
            return Some(nodes.Limit(lhs.inner, min(rhs.n, lhs.n)))
        case nodes.Limit() as lhs, nodes.Slice() as rhs:
            return _merge_limit_then_slice(lhs, rhs)
        case nodes.Slice() as lhs, nodes.Slice() as rhs:
            return _merge_slices(lhs, rhs)
        case nodes.Slice() as lhs, nodes.Limit() as rhs:
            return _merge_slice_then_limit(lhs, rhs)
        case nodes.Sort() as lhs, nodes.Sort() as rhs:
            return Some(lhs)
        case _:
            return NONE


def _merge_limit_then_slice(lhs: nodes.Limit, rhs: nodes.Slice) -> NewNode:
    if rhs.offset < 0:
        return NONE

    available = max(lhs.n - rhs.offset, 0)
    match rhs.length:
        case Some(rhs_length):
            merged = nodes.Slice(
                lhs.inner, Some(min(rhs_length, available)), rhs.offset
            )
            return Some(merged)
        case Null():
            merged = nodes.Slice(lhs.inner, Some(available), rhs.offset)
            return Some(merged)


def _merge_slices(lhs: nodes.Slice, rhs: nodes.Slice) -> Option[nodes.Node]:
    if lhs.offset < 0 or rhs.offset < 0:
        return NONE

    offset = lhs.offset + rhs.offset
    match lhs.length:
        case Some(lhs_length):
            match rhs.length:
                case Some(rhs_length):
                    merged_bounded_slice = nodes.Slice(
                        lhs.inner,
                        Some(min(rhs_length, max(lhs_length - rhs.offset, 0))),
                        offset,
                    )
                    return Some(merged_bounded_slice)
                case Null():
                    merged_open_slice = nodes.Slice(
                        lhs.inner, Some(max(lhs_length - rhs.offset, 0)), offset
                    )
                    return Some(merged_open_slice)
        case Null():
            match rhs.length:
                case Some(rhs_length):
                    rhs_slice = nodes.Slice(lhs.inner, Some(rhs_length), offset=offset)
                    return Some(rhs_slice)
                case Null():
                    unbounded_slice = nodes.Slice(lhs.inner, length=NONE, offset=offset)
                    return Some(unbounded_slice)


def _merge_slice_then_limit(lhs: nodes.Slice, rhs: nodes.Limit) -> NewNode:
    if lhs.offset < 0:
        return NONE

    match lhs.length:
        case Some(lhs_length):
            merged = nodes.Slice(lhs.inner, Some(min(lhs_length, rhs.n)), lhs.offset)
            return Some(merged)
        case Null():
            merged = nodes.Slice(lhs.inner, Some(rhs.n), lhs.offset)
            return Some(merged)


def _merge_filters(lhs: nodes.Filter, rhs: nodes.Filter) -> nodes.Filter:
    predicates = (
        try_iter(lhs.predicates)
        .chain(lhs.more_predicates)
        .chain(_constraints_to_predicates(lhs.constraints))
        .chain(try_iter(rhs.predicates))
        .chain(rhs.more_predicates)
    )
    return nodes.Filter(lhs.inner, predicates, (), rhs.constraints)


def _constraints_to_predicates(
    constraints: Dict[str, IntoExpr],
) -> Iter[IntoExprColumn]:
    from .._funcs import col

    return constraints.items().iter().map_star(lambda key, value: col(key).eq(value))


def _merge_drops(lhs: nodes.Drop, rhs: nodes.Drop) -> nodes.Drop:
    columns = (
        try_iter(lhs.columns)
        .chain(lhs.more_columns)
        .chain(try_iter(rhs.columns))
        .chain(rhs.more_columns)
    )
    return nodes.Drop(lhs.inner, columns, ())


def _merge_renames(lhs: nodes.Rename, rhs: nodes.Rename) -> nodes.Rename:
    names = Iter(lhs.mapping.keys()).chain(rhs.mapping.keys()).collect()
    mapping = (
        names
        .iter()
        .map(
            lambda name: (
                name,
                rhs.mapping.get(
                    lhs.mapping.get(name, name), lhs.mapping.get(name, name)
                ),
            )
        )
        .filter_star(lambda name, renamed: renamed != name)
        .collect(Dict)
    )
    return nodes.Rename(lhs.inner, mapping)
