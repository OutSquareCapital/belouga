from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

from pyochain import NONE, Err, Iter, NoneOption as Null, Ok, Option, Result, Seq, Some

from ._expr import Expr
from ._funcs import col

if TYPE_CHECKING:
    from pyochain.traits import PyoCollection
    from sqlglot import exp

    from ._frame import LazyFrame
    from .typing import JoinStrategy
type OptSeq = Option[Seq[str]]
type JoinKeysRes[T: Seq[str] | str] = Result[JoinKeys[T], ValueError]


@dataclass(slots=True)
class JoinBuilder:
    suffix: str
    left: PyoCollection[str]
    right: PyoCollection[str]

    def equals(self, left: str, right: str) -> Expr:
        return self.lhs(left).eq(self.rhs(right))

    @staticmethod
    def lhs(name: str) -> Expr:
        return col(name, table="lhs")

    @staticmethod
    def rhs(name: str) -> Expr:
        return col(name, table="rhs")

    def for_inner_left(self, name: str) -> Option[Expr]:
        match (name in self.left, name in self.right):
            case (_, True):
                return NONE
            case (True, False):
                return Some(self._aliased(name))
            case _:
                return Some(self.rhs(name))

    def for_outer(self, name: str) -> Expr:
        if name in self.left:
            return self._aliased(name)
        return self.rhs(name)

    def for_right(self, name: str) -> Expr:
        match (name in self.left, name in self.right):
            case (True, False):
                return self._aliased(name)
            case _:
                return self.rhs(name)

    def _aliased(self, name: str) -> Expr:
        return self.rhs(name).alias(f"{name}{self.suffix}")

    def get_join_cols_cross(self) -> Iter[exp.Expr]:
        return (
            self.left
            .iter()
            .map(self.lhs)
            .chain(self.right.iter().map(self.for_outer))
            .map(lambda c: c.inner)
        )

    def get_join_cols(
        self, other: LazyFrame, join_keys: JoinKeys[Seq[str]], how: JoinStrategy
    ) -> Iter[exp.Expr | str]:
        left = self.left.iter()
        right = other.columns.iter()
        match how:
            case "inner" | "left":
                return (
                    left
                    .map(self.lhs)
                    .chain(right.filter_map(self.for_inner_left))
                    .map(lambda c: c.inner)
                )
            case "outer":
                return (
                    left
                    .map(self.lhs)
                    .chain(right.map(self.for_outer))
                    .map(lambda c: c.inner)
                )
            case "right":
                return (
                    left
                    .filter(lambda name: name not in join_keys.left)
                    .map(self.lhs)
                    .chain(right.map(self.for_right))
                    .map(lambda c: c.inner)
                )
            case "semi" | "anti":
                return Iter.once("lhs.*")


class JoinKeys[T: Seq[str] | str](NamedTuple):
    left: T
    right: T

    @staticmethod
    def from_on(
        on: Option[str], left_on: Option[str], right_on: Option[str]
    ) -> JoinKeysRes[str]:
        match (on, left_on, right_on):
            case (Some(on_key), Null(), Null()):
                return Ok(JoinKeys(on_key, on_key))
            case (Null(), Some(lk), Some(rk)):
                return Ok(JoinKeys(lk, rk))
            case (Null(), _, _):
                msg = "Either (`left_on` and `right_on`) or `on` keys should be specified."
                return Err(ValueError(msg))
            case _:
                msg = "If `on` is specified, `left_on` and `right_on` should be None."
                return Err(ValueError(msg))

    @staticmethod
    def from_by(by: OptSeq, by_left: OptSeq, by_right: OptSeq) -> JoinKeysRes[Seq[str]]:
        match (by, by_left, by_right):
            case (Some(vals), Null(), Null()):
                return Ok(JoinKeys(vals, vals))
            case (Null(), Some(left_vals), Some(right_vals)):
                if left_vals.length() == right_vals.length():
                    return Ok(JoinKeys(left_vals, right_vals))
                msg = "`by_left` and `by_right` must have the same length."
                return Err(ValueError(msg))
            case (Null(), Null(), Null()):
                empty = Seq[str].new()
                return Ok(JoinKeys(empty, empty))
            case (Null(), _, _):
                msg = "Can not specify only `by_left` or `by_right`, you need to specify both."
                return Err(ValueError(msg))
            case _:
                msg = "If `by` is specified, `by_left` and `by_right` should be None."
                return Err(ValueError(msg))

    @staticmethod
    def from_how(
        how: JoinStrategy, on: OptSeq, left_on: OptSeq, right_on: OptSeq
    ) -> JoinKeysRes[Seq[str]]:
        match (on, left_on, right_on):
            case (Some(on_vals), Null(), Null()):
                return Ok(JoinKeys(on_vals, on_vals))
            case (Null(), Some(lv), Some(rv)) if lv.length() == rv.length():
                return Ok(JoinKeys(lv, rv))
            case (Null(), Some(_), Some(_)):
                msg = "`left_on` and `right_on` must have the same length."
                return Err(ValueError(msg))
            case (Some(_), _, _):
                msg = f"If `on` is specified, `left_on` and `right_on` should be None for {how}."
                return Err(ValueError(msg))
            case _:
                msg = f"Either (`left_on` and `right_on`) or `on` keys should be specified for {how}."
                return Err(ValueError(msg))

    def get_join_condition(self: JoinKeys[Seq[str]], builder: JoinBuilder) -> exp.Expr:
        return (
            self.left
            .iter()
            .zip(self.right)
            .map_star(builder.equals)
            .reduce(Expr.and_)
            .inner
        )
