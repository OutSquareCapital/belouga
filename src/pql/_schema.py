from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import pyochain as pc

from . import sql
from ._datatypes import DataType


@dataclass(slots=True, init=False)
class Schema(pc.Dict[str, DataType]):
    @classmethod
    def from_frame(cls, frame: sql.Relation) -> Self:
        dtypes = frame.dtypes.iter().map(
            lambda d: DataType.__from_sql__(sql.DType.parse(d))
        )
        return cls(frame.columns.iter().zip(dtypes, strict=True))
