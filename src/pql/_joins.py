from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, NamedTuple

import pyochain as pc

from . import sql

if TYPE_CHECKING:
    from pyochain.traits import PyoCollection

    from ._typing import JoinKeysRes, JoinStrategy
    from .sql import Expr
type OptSeq = pc.Option[pc.Seq[str]]

_RHS = partial(sql.col, table="rhs")
_LHS = partial(sql.col, table="lhs")


@dataclass(slots=True)
class JoinBuilder:
    suffix: str
    left: PyoCollection[str]
    right: PyoCollection[str]

    def equals(self, left: str, right: str) -> Expr:
        return self.lhs(left).eq(self.rhs(right))

    @staticmethod
    def lhs(name: str) -> Expr:
        return _LHS(name)

    def for_inner_left(self, name: str) -> pc.Option[Expr]:
        match (name in self.left, name in self.right):
            case (_, True):
                return pc.NONE
            case (True, False):
                return pc.Some(self._aliased(name))
            case _:
                return pc.Some(self.rhs(name))

    def for_outer(self, name: str) -> Expr:
        match name in self.left:
            case True:
                return self._aliased(name)
            case False:
                return self.rhs(name)

    def for_right(self, name: str) -> Expr:
        match (name in self.left, name in self.right):
            case (True, False):
                return self._aliased(name)
            case _:
                return self.rhs(name)

    def _aliased(self, name: str) -> Expr:
        return self.rhs(name).alias(f"{name}{self.suffix}")

    @staticmethod
    def rhs(name: str) -> Expr:
        return _RHS(name)


class JoinKeys[T: pc.Seq[str] | str](NamedTuple):
    left: T
    right: T

    @staticmethod
    def from_on(
        on: pc.Option[str], left_on: pc.Option[str], right_on: pc.Option[str]
    ) -> JoinKeysRes[str]:
        match (on, left_on, right_on):
            case (pc.Some(on_key), pc.NONE, pc.NONE):
                return pc.Ok(JoinKeys(on_key, on_key))
            case (pc.NONE, pc.Some(lk), pc.Some(rk)):
                return pc.Ok(JoinKeys(lk, rk))
            case (pc.NONE, _, _):
                msg = "Either (`left_on` and `right_on`) or `on` keys should be specified."
                return pc.Err(ValueError(msg))
            case _:
                msg = "If `on` is specified, `left_on` and `right_on` should be None."
                return pc.Err(ValueError(msg))

    @staticmethod
    def from_by(
        by: OptSeq, by_left: OptSeq, by_right: OptSeq
    ) -> JoinKeysRes[pc.Seq[str]]:
        match (by, by_left, by_right):
            case (pc.Some(vals), pc.NONE, pc.NONE):
                return pc.Ok(JoinKeys(vals, vals))
            case (pc.NONE, pc.Some(left_vals), pc.Some(right_vals)):
                match left_vals.length() == right_vals.length():
                    case True:
                        return pc.Ok(JoinKeys(left_vals, right_vals))
                    case False:
                        msg = "`by_left` and `by_right` must have the same length."
                        return pc.Err(ValueError(msg))
            case (pc.NONE, pc.NONE, pc.NONE):
                fn = pc.Seq[str].new
                return pc.Ok(JoinKeys(fn(), fn()))
            case (pc.NONE, _, _):
                msg = "Can not specify only `by_left` or `by_right`, you need to specify both."
                return pc.Err(ValueError(msg))
            case _:
                msg = "If `by` is specified, `by_left` and `by_right` should be None."
                return pc.Err(ValueError(msg))

    @staticmethod
    def from_how(
        how: JoinStrategy, on: OptSeq, left_on: OptSeq, right_on: OptSeq
    ) -> JoinKeysRes[pc.Seq[str]]:
        match (on, left_on, right_on):
            case (pc.Some(on_vals), pc.NONE, pc.NONE):
                return pc.Ok(JoinKeys(on_vals, on_vals))
            case (pc.NONE, pc.Some(lv), pc.Some(rv)) if lv.length() == rv.length():
                return pc.Ok(JoinKeys(lv, rv))
            case (pc.NONE, pc.Some(_), pc.Some(_)):
                msg = "`left_on` and `right_on` must have the same length."
                return pc.Err(ValueError(msg))
            case (pc.Some(_), _, _):
                msg = f"If `on` is specified, `left_on` and `right_on` should be None for {how}."
                return pc.Err(ValueError(msg))
            case _:
                msg = f"Either (`left_on` and `right_on`) or `on` keys should be specified for {how}."
                return pc.Err(ValueError(msg))
