from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, NamedTuple

from pyochain import NONE, Err, NoneOption as Null, Ok, Option, Seq, Some

from ._funcs import col

if TYPE_CHECKING:
    from pyochain.traits import PyoCollection

    from ._expr import Expr
    from .typing import JoinKeysRes, JoinStrategy
type OptSeq = Option[Seq[str]]

_RHS = partial(col, table="rhs")
_LHS = partial(col, table="lhs")


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

    def for_inner_left(self, name: str) -> Option[Expr]:
        match (name in self.left, name in self.right):
            case (_, True):
                return NONE
            case (True, False):
                return Some(self._aliased(name))
            case _:
                return Some(self.rhs(name))

    def name_for_inner_left(self, name: str) -> Option[str]:
        match (name in self.left, name in self.right):
            case (_, True):
                return NONE
            case (True, False):
                return Some(f"{name}{self.suffix}")
            case _:
                return Some(name)

    def for_outer(self, name: str) -> Expr:
        match name in self.left:
            case True:
                return self._aliased(name)
            case False:
                return self.rhs(name)

    def name_for_outer(self, name: str) -> str:
        return f"{name}{self.suffix}" if name in self.left else name

    def for_right(self, name: str) -> Expr:
        match (name in self.left, name in self.right):
            case (True, False):
                return self._aliased(name)
            case _:
                return self.rhs(name)

    def name_for_right(self, name: str) -> str:
        match (name in self.left, name in self.right):
            case (True, False):
                return f"{name}{self.suffix}"
            case _:
                return name

    def _aliased(self, name: str) -> Expr:
        return self.rhs(name).alias(f"{name}{self.suffix}")

    @staticmethod
    def rhs(name: str) -> Expr:
        return _RHS(name)


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
                match left_vals.length() == right_vals.length():
                    case True:
                        return Ok(JoinKeys(left_vals, right_vals))
                    case False:
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
