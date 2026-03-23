from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, ClassVar

from ._code_gen import Relation
from ._creation import into_relation

if TYPE_CHECKING:
    from duckdb import DuckDBPyRelation

    from .typing import IntoRel, Orientation


class Frame(Relation):
    _inner: DuckDBPyRelation
    __slots__: ClassVar[Iterable[str]] = ("_inner",)

    def __init__(self, data: IntoRel, orient: Orientation = "col") -> None:
        self._inner = into_relation(data, orient=orient)
